import pandas as pd
from influxdb import InfluxDBClient
from datetime import datetime, timedelta
import sys
from typing import List, Dict, Any, Optional
from details import get_journal_metrics, calculate_group_distances
from fastapi.encoders import jsonable_encoder
import constant as cs

# --- InfluxDB Client Setup ---
INFLUXDB_HOST = cs.INFLUXDB_HOST
INFLUXDB_PORT = cs.INFLUXDB_PORT
INFLUXDB_DATABASE = cs.INFLUXDB_DATABASE

try:
    client = InfluxDBClient(host=INFLUXDB_HOST, port=INFLUXDB_PORT, database=INFLUXDB_DATABASE)
    client.ping()
except Exception as e:
    print(f"Error connecting to InfluxDB: {e}", file=sys.stderr)
    client = None


class Journal:
    def __init__(self, query: str) -> None:
        self.query = query
        self.result = None
        self.points: Optional[List[Dict[str, Any]]] = None
        self.df = pd.DataFrame()
        self.is_empty = True

        if client is None:
            return

        try:
            self.setResult()
            self.setPoints()
            self.setDf()
            self.checkIsEmpty()
        except Exception as e:
            print(f"Error during Journal init: {e}", file=sys.stderr)

    def setResult(self):
        if client:
            self.result = client.query(query=self.query)

    def setPoints(self):
        if self.result:
            self.points = list(self.result.get_points())
        else:
            self.points = []

    def setDf(self):
        if self.points:
            self.df = pd.DataFrame(self.points)
        else:
            self.df = pd.DataFrame()

    def checkIsEmpty(self):
        self.is_empty = self.df.empty

# summary = {EWH : 0, stopCount : 0, stopDur : 0, travelDist : 0, travelDur : 0};
def convert_date(date):
    return datetime.fromisoformat(date)

def generate_journal_statistics_logic(cbox_input, start_date_str, end_date_str, threshold_input, is_detailed):
    MAX_DIFF_SEND = (30 * 60)
    condition = f"CBOX = '{cbox_input}'"
    start_date = datetime.strptime(start_date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    prev_date = start_date - timedelta(days=1)
    FIRST_POSITION = Journal(
        f"SELECT time, EV, SPD, CBOX, LAT, LON, ATTRIBUTES FROM journal WHERE time < '{start_date}' AND time > '{prev_date}' AND {condition} ORDER BY time DESC LIMIT 1"
    ).df
    # FIRST_POSITION['time'] = pd.to_datetime(FIRST_POSITION['time'], errors='coerce', utc=True)
    # print(FIRST_POSITION)
    print(f"SELECT time, EV, SPD, CBOX, LAT, LON, ATTRIBUTES FROM journal WHERE time >='{start_date_str}' AND time <='{end_date_str}' AND {condition}")
    veh = Journal(f"SELECT time, EV, SPD, CBOX, LAT, LON, ATTRIBUTES FROM journal WHERE time >='{start_date_str}' AND time <='{end_date_str}' AND {condition}")
    df = pd.concat([FIRST_POSITION, veh.df], ignore_index=True)
    df['time'] = df['time'].apply(convert_date)
    # Preprocessing time
    df = df.dropna(subset=['time'])
    df['diff'] = df['time'].diff()
    mask_diff = df['diff'].dt.total_seconds() > MAX_DIFF_SEND
    df.loc[mask_diff, 'EV'] = 2

    distanceMethodeCalcType = 0 
    current_time = pd.Timestamp.now(tz='UTC')
    if end_date.date() < current_time.date():
        distanceMethodeCalcType = 1
    else:
        distanceMethodeCalcType = 2
    # extract values of old days  from the db or calc it if current day inclus
    data = get_journal_metrics(df[['time','CBOX','EV','LAT','LON','SPD']].copy(),start_date, end_date, distanceMethodeCalcType, threshold_input)
    print("summart " , data)
    if not is_detailed:
        df.loc[df['EV'] == 4, 'EV'] = 2
        df.loc[df['EV'] == 3, 'EV'] = 1
    
    # Group & aggregation
    ev_changes = df['EV'] != df['EV'].shift()
    df['group'] = ev_changes.cumsum()
    group_stats = df.groupby('group').agg({
        'EV': 'first',
        'time': 'first',
        'CBOX': 'first',
        'SPD': ['mean', 'max'],
        'ATTRIBUTES': 'first',
        'LAT': 'first',
        'LON': 'first'
    })
    
    
    # Summary object
    SUMMARY = {
        "EWH": float(data['engineWorkingHours']) / 24,
        "stopCount": int(data['stopCount'] ),
        "stopDur": float(data['stopDuration']) / 24,
        "travelDist": float(data['DailyDistance']),
        "travelDur": float(data['travelHours']) / 24
    }
    
    group_stats.columns = ['EV', 'DT', 'CBOX', 'SPDAVG', 'SPDMAX', 'ATTRIBUTES', 'LAT', 'LON']
    group_stats['AGE'] = group_stats['DT'].shift(-1) - group_stats['DT']
    group_stats['AGE'] = group_stats['AGE'].dt.total_seconds() / (3600 * 24)

    # Distance calculation
    group_stats['DIST'] = calculate_group_distances(df.copy())
    group_stats['DIST'].fillna(0, inplace=True)

    # Add constant columns
    group_stats['EWH'] = 0
    group_stats['CWP'] = -1
    group_stats['LWP'] = None
    group_stats['CTWP'] = None

    # Final AGE fix
    end_time = pd.to_datetime(end_date_str)
    end_time = end_time.tz_localize('UTC') if end_time.tzinfo is None else end_time.tz_convert('UTC')
    last_position_time = min(current_time, end_time) - group_stats['DT'].iloc[-1]
    group_stats.at[group_stats.index[-1], 'AGE'] = last_position_time.total_seconds() / (3600 * 24)

    # Return final result
    if threshold_input in [0, 1]:
        group_stats['DT'] = group_stats['DT'].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        journal = group_stats[['CBOX', 'EV', 'DT', 'AGE', 'DIST', 'SPDAVG', 'SPDMAX', 'ATTRIBUTES', 'LAT', 'LON', 'EWH', 'CWP', 'LWP', 'CTWP']].to_dict(orient='records')
        res = {"journal": journal, "summary": SUMMARY}
        return jsonable_encoder(res)

    # Advanced filtering
    mask = ~((group_stats['AGE'] < threshold_input / (3600 * 24)) & (group_stats['EV'].isin([2, 4])))
    filtered_stats = group_stats[mask].copy()
    ev_changes = filtered_stats['EV'] != filtered_stats['EV'].shift()
    filtered_stats['group_2'] = ev_changes.cumsum()
    wave_two_group = filtered_stats.groupby('group_2').agg({
        'CBOX': 'first',
        'EV': 'first',
        'DT': 'first',
        'AGE': 'first',
        'ATTRIBUTES': 'first',
        'DIST': 'sum',
        'SPDAVG': 'mean',
        'SPDMAX': 'max',
        'LAT': 'first',
        'LON': 'first',
        'EWH': 'first',
        'CWP': 'first',
        'LWP': 'first',
        'CTWP': 'first',
    })
    wave_two_group.columns = ['CBOX', 'EV', 'DT', 'AGE', 'ATTRIBUTES', 'DIST', 'SPDAVG', 'SPDMAX', 'LAT', 'LON', 'EWH', 'CWP', 'LWP', 'CTWP']
    wave_two_group['AGE'] = wave_two_group['DT'].shift(-1) - wave_two_group['DT']
    wave_two_group.at[wave_two_group.index[-1], 'AGE'] = last_position_time
    wave_two_group['AGE'] = wave_two_group['AGE'].dt.total_seconds() / (3600 * 24)
    nbArreit = 0
    if not is_detailed:
        nbArreit = wave_two_group[wave_two_group['EV'] == 2].shape[0]
    else:
        # TODO: wil be fixed later 
        if wave_two_group['EV'].iloc[-1] == 2 and wave_two_group['EV'].iloc[-2] == 4:
            nbArreit = wave_two_group[wave_two_group['EV'] == 1].shape[0] + 2
        else:
            def calc_arreit(current_ev, next_ev):
                if (current_ev in [2, 4]) and (next_ev in [2, 4] or next_ev == 1):
                    return True
                return False
            wave_two_group['next_ev'] = wave_two_group['EV'].shift(-1)   
            wave_two_group['nb_arreit'] = wave_two_group.apply(
                lambda row: calc_arreit(row['EV'], row['next_EV']),
                axis=1
            )
            nbArreit = wave_two_group[wave_two_group['nb_arreit'] == True].shape[0]

        # for i in range(len(group_stats) - 1):
        #     current_ev = group_stats['EV'].iloc[i]
        #     next_ev = group_stats['EV'].iloc[i + 1]
        #     if current_ev in [2, 4] and next_ev in [2, 4]:
        #         nbArreit += 1
        #     elif current_ev in [2, 4] and next_ev == 1:
        #         nbArreit += 1
    SUMMARY = {
        "EWH": float(data['engineWorkingHours']) / 24,
        "stopCount": nbArreit,
        "stopDur": float(data['stopDuration']) / 24,
        "travelDist": float(data['DailyDistance']),
        "travelDur": float(data['travelHours']) / 24
    }
    # print(wave_two_group[['EV','DT','AGE']].head(10))
    # print(" ev stats  ", wave_two_group['EV'].value_counts())
    journal = wave_two_group[['CBOX', 'EV', 'DT', 'AGE', 'ATTRIBUTES', 'DIST', 'SPDAVG', 'SPDMAX', 'LAT', 'LON', 'EWH', 'CWP', 'LWP', 'CTWP']].to_dict(orient='records')
    res = {"journal": journal, "summary": SUMMARY}

    return jsonable_encoder(res)