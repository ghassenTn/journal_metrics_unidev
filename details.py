from datetime import timedelta
import json
import pandas as pd
import numpy as np
import sys
import fdb
import warnings
import math
sys.setrecursionlimit(10000000)
warnings.filterwarnings("ignore")
import constant as cs

## Firebird database connection details

host = cs.host
user = cs.user
password = cs.password
port = cs.password
con_str = cs.con_str
connection = fdb.connect(con_str, user=user, password=password)
cursor = connection.cursor()


# get specific data from db
def getIdEnergieTypeFromVehTable(cbox):
    cursor = connection.cursor()
    try:
        query = f"select ID_ENERGY_TYPE from VEH where cbox = {cbox}"
        return cursor.execute(query).fetchone()[0]
    except:
        return -1
    

def getFuelType(cbox):
    cursor = connection.cursor()
    try:
        id_energie_type = getIdEnergieTypeFromVehTable(cbox)
        query = f"select COST from ENERGY_TYPE where ID_ENERGY_TYPE = {id_energie_type}"
        return cursor.execute(query).fetchone()[0]
    except:
        return -1

# def get_consumption_rate(cbox):
#     cursor = connection.cursor()
#     query= f" select CONSOMATION_MOY from VEH where cbox = {cbox}" 
#     consumption_rate = cursor.execute(query)
#     return consumption_rate.fetchone()


# calculate the score 
def calculate_score(distance_value, nbhardbraking, nbhardacceleration, templiquide, travelHours, maxTravelHours, NombreFatime, maxspeed, averageSpeed):
    distance_intervals = [
        {"type": "court", "intervalle": '0-40', "weights": {'nbhardbraking': -5, 'nbhardacceleration': -5, 'templiquide': 0.6, 'travelHours': 0.3, 'maxTravelHours': 0.1, 'NombreFatime': 1, 'maxspeed': -0.01, 'averageSpeed': 0.1}},
        {"type": "medium", "intervalle": '41-100', "weights": {'nbhardbraking': -4, 'nbhardacceleration': -4, 'templiquide': 0.6, 'travelHours': 0.2, 'maxTravelHours': 0.05, 'NombreFatime': -1, 'maxspeed': -0.01, 'averageSpeed': 0.1}},
        {"type": "long", "intervalle": '101-300', "weights": {'nbhardbraking': -2, 'nbhardacceleration': -2, 'templiquide': 0.6, 'travelHours': 0.07, 'maxTravelHours': 0.05, 'NombreFatime': -1, 'maxspeed': -0.01, 'averageSpeed': 0.1}},
        {"type": "x-long", "intervalle": '301-2000', "weights": {'nbhardbraking': -1, 'nbhardacceleration': -1, 'templiquide': 0.6, 'travelHours': 0.02, 'maxTravelHours': 0.03, 'NombreFatime': 0, 'maxspeed': -0.01, 'averageSpeed': 0.1}},
    ]
    
    weights = next((interval['weights'] for interval in distance_intervals if float(interval['intervalle'].split('-')[0]) <= distance_value <= float(interval['intervalle'].split('-')[1])), None)
    if not weights:
        score = -1
        return score
    try:
        score = 0
        score += nbhardbraking * weights['nbhardbraking'] if nbhardacceleration > 0 else 3
        score += nbhardacceleration * weights['nbhardacceleration'] if nbhardbraking > 0 else 3
        score += templiquide * weights['templiquide'] if isinstance(templiquide, (int, float)) and not np.isnan(templiquide) else 30 * weights['templiquide']
        score += min(travelHours, 6) * weights['travelHours']
        score += min(maxTravelHours, 3) * weights['maxTravelHours']
        score += NombreFatime * weights['NombreFatime'] if NombreFatime <= 3 else NombreFatime * -1
        score += maxspeed * weights['maxspeed'] if maxspeed > 130 else 10
        score += min(averageSpeed, 66.5) * weights['averageSpeed']
        return score
    except:
        score = -1
        return score
        
#Nov 25 2024 
# def count_stops(df, threshold=60):  
#     nb_arret = 0  
#     arret_intervalle = []
#     en_arret = 0
#     for index, row in df.iterrows():
#         if (row['EV'] == 2 or row['EV'] == 4):  
#             arret_intervalle.append(row['time'])  
#         else:  
#             if arret_intervalle:
#                 time_diff = (arret_intervalle[-1] - arret_intervalle[0]).total_seconds()  
#                 if time_diff >= threshold: 
#                     nb_arret += 1
#                 arret_intervalle = []
#                 en_arret += time_diff
#     if arret_intervalle:  
#         time_diff = (arret_intervalle[-1] - arret_intervalle[0]).total_seconds()  
#         if time_diff >= threshold:
#             nb_arret += 1  
#         en_arret += time_diff
#     return nb_arret


def totaleLitreFilled(df):
    """
    Calculate the total liters of fuel filled.
    Parameters:
    df (DataFrame): The input DataFrame.
    Returns:
    float: The total liters of fuel filled.
    0  : If the vehicule dont have io84 sensor. 
    """
    
    if df['io84'].isnull().all():
        return 0
    try:
        maxDiff = df['io84'].diff().max() / 10
        return maxDiff if maxDiff > 20 else 0
    except:
        return 0

def getNbFatTime(df):
    try:
        ft = 0.4     
        df['time_diff'] = df['time'].diff().dt.total_seconds().fillna(0)        
        df['is_running'] = (df['SPD'] > 0)
        df['running_change'] = df['is_running'].astype(int).diff().fillna(0)        
        df['running_group'] = df['running_change'].abs().cumsum() * df['is_running']        
        running_times = df.groupby('running_group')['time_diff'].sum().div(60)
        running_times = running_times[running_times.index != 0]
        fatTime = (running_times.div(60) >= ft).sum()     
        max_running_time = running_times.max() if not running_times.empty else 0
        return max_running_time / 100, fatTime
    except Exception as e:
        # print(f"Error calculating number of fatigued times: {e}")
        return 0, 0

def fuelCoast(cbox,daily_fuel):
    if daily_fuel == 0:
        return 0
    fuel_type = getFuelType(cbox)
    try:
        if daily_fuel >= 0 and fuel_type >= 0:
            return daily_fuel * getFuelType(cbox)
        return 0
    except:
        return 0

def getStartTime(df):
    """
     Get the start time of the trip.

        Parameters:
        df (DataFrame): The input DataFrame.

        Returns:
        datetime: The start time of the trip.
    """
    
    try:
        return [row['time'] for _,row in df.iterrows(   ) if row['SPD'] > 0][0]
    except:
        return df['time'].iloc[0] if df.shape[0] > 0 else "2024-01-01 00:00:00"

def getEndTime(df:pd.DataFrame)->pd.Timestamp:
    """
     Get the end time of the trip.
        Parameters:
        df (DataFrame): The input DataFrame.
        Returns:
        datetime: The end time of the trip.
    """
    try:
        return [row['time'] for _,row in df.iterrows() if row['SPD'] > 0 ][-1]
    except:
        return df['time'].iloc[0] if df.shape[0] > 0 else "2024-01-01 00:00:00"


def reffile_number(df:pd.DataFrame)->float:
    try:
        if df['io84'].isnull().all():
            return 0
        return df[df['io84'].diff()/10> 20].shape[0]
    except:
        return 0

# Mar 21 2025
def calculate_distance(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return 0
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of Earth in kilometers
    r = 6371.0
    return c * r

# Mar 28 2025
def get_consumption_rate_db(cbox):
    qr = f"""
        select v.CONSOMATION_MOY_BY_SPD from veh v where v.CBOX = {cbox}
    """
    return connection.cursor().execute(qr).fetchone()[0]
# print(get_consumption_rate_db(1014))
# Mar 21 2025        
def get_consumption_rate(speed,speed_consumption):
    if speed < 1:
        return 0
    for row in speed_consumption:
        if row['min_speed'] <= speed <= row['max_speed']:
            return row['fuel_rate']
    return 8.4

def get_consumption_rate_engine_agricol(speed,speed_consumption):
    if speed < 1:
        return 0
    for row in speed_consumption:
        if row['min_speed'] <= speed <= row['max_speed']:
            return row['fuel_rate'] 
    return 8.4

# Dec 18 2024
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> np.array:
    """
    Calculate the great circle distance in kilometers between two points
    on the Earth (specified in decimal degrees).
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])        
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))        
    km = 6371 * c
    # Avoid outliers with threshold 10 km
    return km

def allInOne(df: pd.DataFrame, threshold=60, correction_factor=1.015) -> tuple:
    df_f = df.copy()
    # NB_DAYS = df['time'].dt.date().nunique()
    # print(NB_DAYS)
    next_df = df.shift(-1)
    df["prev_LAT"] = df["LAT"].shift(1)
    df["prev_LON"] = df["LON"].shift(1)    
    time_diff = next_df['time'] - df['time']
    df['time_diff'] = df['time'].diff().dt.total_seconds()
    df['is_stop'] = df['EV'].isin([2, 4])
    df['stop_group'] = (df['is_stop'] != df['is_stop'].shift()).cumsum()
    
    mask_travel_hours = (
        (df['EV'] == 1) & 
        (df['SPD'] > 0) & 
        (next_df['EV'] == 1) & 
        (next_df['SPD'] > 0)  
    )
    mask_distance = (df['SPD'] > 0) & (next_df['SPD'] > 0)
    mask_working_engine_hours = (df['ignition'] == True) 
    mask_regulator_vitesse = (df['SPD'] > 50) & ((next_df['time'] - df['time']).dt.total_seconds() < 60)
    mask_en_arreit = ((df['EV'] == 2) | (df['EV'] == 4) )
    lat1 = df['LAT'][mask_distance].to_numpy()
    lon1 = df['LON'][mask_distance].to_numpy()
    lat2 = next_df['LAT'][mask_distance].to_numpy()
    lon2 = next_df['LON'][mask_distance].to_numpy()            
    total_distance = 0
    distances = haversine(lat1, lon1, lat2, lon2)

    #  # Check if 'odometer' is available for total distance calculation or pass to io87
    # if 'odometer' in df.columns and df['odometer'].notna().sum() > 0:
    #     if df['odometer'].iloc[0] > 0 and df['odometer'].iloc[-1] > 0:
    #         total_distance = (df['odometer'].iloc[-1] - df['odometer'].iloc[0])/1000   
    #  # Check if 'io87' is available for total distance calculation      
    # if 'io87' in df.columns and df['io87'].notna().sum() > 0:
    #     if df['io87'].iloc[0] > 0 and df['io87'].iloc[-1] > 0:
    #         total_distance = (df.iloc[-1]['io87'] - df.iloc[0]['io87'])/1000
    # else:
        # Fall back to Haversine calculation if 'io87' and 'odometer' is unavailable or pass to haversine formule
    try:
        total_distance = distances.sum()
    except:
        total_distance = 0
    
    stop_groups = df[df['is_stop']].groupby('stop_group')['time_diff'].sum()
    
    try:
        total_time_seconds_travel_hours = time_diff[mask_travel_hours].dt.total_seconds().sum()
    except:
        total_time_seconds_travel_hours = 0
    try:    
        total_time_working_engine_hours = time_diff[mask_working_engine_hours].dt.total_seconds().sum()
    except:
        total_time_working_engine_hours = 0
    try:
        total_time_regualtor_vitesse = time_diff[mask_regulator_vitesse].dt.total_seconds().sum()
    except:
        total_time_regualtor_vitesse = 0
    try:
        total_arreit = time_diff[mask_en_arreit].dt.total_seconds().sum()
    except:
        total_arreit = 0
    try:
        nb_arret = (stop_groups >= threshold).sum()
    except:
        nb_arret = 0

    if total_time_working_engine_hours >= total_time_seconds_travel_hours:
        total_time_seconds_idle_hours = total_time_working_engine_hours  - total_time_seconds_travel_hours
    else:
        total_time_seconds_idle_hours = 0
    
    total_time_en_arreit =total_arreit

    if total_distance == 0:
        dailyFuel = 0
    else:
        # if not df['io84'].isnull().all():
        #     df_sensor = df.sort_values(by='time').copy()
        #     cleaned_rows = []
        #     prev_level = None
        #     for _, row in df_sensor.iterrows():
        #         current_level = row['io84']
        #         if prev_level is None or current_level <= prev_level:
        #             cleaned_rows.append(row)
        #             prev_level = current_level
        #     cleaned_df = pd.DataFrame(cleaned_rows)
        #     dailyFuel = (cleaned_df['io84'].iloc[0] - cleaned_df['io84'].iloc[-1])/10
        #     del df_sensor
        #     del cleaned_df
        #     del cleaned_rows
        # # elif not df['io89'].isnull().all():
        # #     dailyFuel = df['io89'].iloc[-1] - df['io89'].iloc[0]
        # else:
            raw_consumption_data = get_consumption_rate_db(int(idBoite))
            if raw_consumption_data is None:
                dailyFuel = 0
            else:
                speed_consumption = json.loads(raw_consumption_data)
                if speed_consumption[0]['rate_type'] == 'per_hour':
                    df_f['consumMoy'] = df_f["SPD"].apply(lambda speed: get_consumption_rate_engine_agricol(speed, speed_consumption))
                    spd_next = df_f['SPD'].shift(-1)
                    time_next = df_f['time'].shift(-1)
                    time_diff = (time_next - df_f['time']).dt.total_seconds()
                    mask = (df_f['SPD'] > 0) & (spd_next > 0) & (time_diff < 300)
                    fuel_consumption = np.where(mask, time_diff * (df_f['consumMoy'] / 3600), 0)
                    dailyFuel = fuel_consumption.sum() + ((total_time_seconds_idle_hours / 3600) * 3.40687)  # Idle state fuel consumption for Agricultural Tractors
                elif speed_consumption[0]['rate_type'] == 'per_100km':
                    df["consumption_rate"] = df["SPD"].apply(lambda speed: get_consumption_rate(speed=speed, speed_consumption=speed_consumption))
                    df['distance_km'] = df.apply(lambda row: calculate_distance(row["prev_LAT"], row["prev_LON"], row["LAT"], row["LON"]), axis=1)
                    df["fuel_used"] = df["distance_km"] * df["consumption_rate"]
                    df['fuel_consum'] = (df["fuel_used"] / 100).cumsum()
                    dailyFuel = df['fuel_consum'].iloc[-1] if not df['fuel_consum'].isna().all() else 0
                    # if speed_consumption[0]['fuel_rate'] == 10:
                    #     dailyFuel += (total_time_seconds_idle_hours / 3600) * 0.64352  # Idle state fuel consumption for Passenger Cars
                    # else:
                    #     dailyFuel += (total_time_seconds_idle_hours / 3600) * 3.02833  # Idle state fuel consumption for Heavy-Duty Trucks
                else:
                    dailyFuel = 0

    return total_time_seconds_idle_hours / 3600, total_time_seconds_travel_hours / 3600, total_distance, total_time_en_arreit / 3600, total_time_working_engine_hours / 3600, total_time_regualtor_vitesse / 3600, dailyFuel, nb_arret


def final(df):
    """
    Calculate the final results based on the input DataFrame.
    Args:
        df (DataFrame): The input DataFrame.

    Returns:
    dict: A dictionary containing the final results.
    """
    global idBoite    
    idBoite = str(df['CBOX'].iloc[0])
    date = df['time'].iloc[0]
    
    idleHours, travelHours, nbkm, stop_duration, work_duration, regulatorVitesse, dailyFuel, stop_count = allInOne(df)

    nbhardbraking = len(df[df['alarm']=='hardBraking'])
    
    nbhardacceleration = len(df[df['alarm']=='hardAcceleration'])
    
    try:
        templiquide = df[df['SPD']>0]['templiquide'].iloc[0]
    except:
        templiquide = 0
    try:
        spd_max = df['SPD'].max()
        maxspeed = spd_max if spd_max < 200 else 100
    except:
        maxspeed = 100

    averageSpeed = df['SPD'].mean() 
    
    startTime = getStartTime(df)
    
    endTime = getEndTime(df)
   
    maxTravelHours ,NombreFatime   = getNbFatTime(df) if nbkm > 0 else (0,0)
    
    try:
        litersFilled = totaleLitreFilled(df['io84']) if df['io84'].notnull().all() else 0
    except:
        litersFilled = totaleLitreFilled(df)
    
    try:
        reffileNumber = reffile_number(df['io84']) if df['io84'].notnull().all() else 0
    except:
        reffileNumber = reffile_number(df)

    try:
        score = float( calculate_score(nbkm,
            nbhardbraking, nbhardacceleration, templiquide, travelHours,
            maxTravelHours, NombreFatime, maxspeed, averageSpeed
        )) 
    except:
        score = 0
    
    try:
        templiquide = float(templiquide)
    except:
        templiquide = 'not available'
    
    return {'date': date,
            'DailyDistance': nbkm.__round__(2),
            'dailyFuel':dailyFuel.__round__(2),
            'litersFilled':litersFilled,
            'refillsNumber':reffileNumber,
            'startTime':startTime,
            'endTime':endTime,
            'travelHours':travelHours.__round__(2),
            'maxTravelHours':maxTravelHours.__round__(2),
            'NombreFatime':NombreFatime,
            'nbHardBraking': nbhardbraking,
            'nbHardAcceleration': nbhardacceleration,
            'IdleHours': idleHours.__round__(2) ,
            'regulatorUsageMinutes':regulatorVitesse.__round__(2),
            'maxSpeed': maxspeed,
            'averageSpeed':averageSpeed.__round__(2),
            'fuel_cost':fuelCoast(idBoite,dailyFuel).__round__(2),
            'score': score.__round__(2),
            "stop_count":stop_count,
            "stop_duration":stop_duration.__round__(2),
            "engine_working_hours":work_duration.__round__(2)
            }
                
def allMetricsCalcResult(df):
    """
    Calculate all the metrics based on the input DataFrame.
    Parameters:
    df (DataFrame): The input DataFrame.
    Returns:
    dict: A dictionary containing all the calculated metrics.
    """
    
    if 'ATTRIBUTES' not in df.columns:
        pass
    else:
        def extract_attributes(attr):
            data = json.loads(attr)
            return (data.get('ignition', None), data.get('motion', None),
                    data.get('coolantTemp', None),
                    data.get('alarm', 'safe'),
                    data.get('io84', None),
                    data.get('io87', None),
                    data.get('power', None),
                    data.get('odometer',None),
                    data.get('hours',None))
        columns = ['ignition', 'motion','templiquide','alarm','io84','io87','power','odometer','hours']
        df[columns] = df['ATTRIBUTES'].apply(extract_attributes).tolist()
        df.drop(['CAP', 'VALID', 'ATTRIBUTES'], axis=1, inplace=True)
        try:
            df['time'] = pd.to_datetime(df['time'])
        except ValueError:
            try:
                df['time'] = pd.to_datetime(df['time'], format='ISO8601')
            except ValueError:
                df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
        return final(df)


def get_working_engine_hours(df) -> float:
    """ return the value of weh """
    df = df.copy()
    try:
        df['time'] = pd.to_datetime(df['time'])
    except ValueError:
        try:
            df['time'] = pd.to_datetime(df['time'], format='ISO8601')
        except ValueError:
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
    def extract_attributes(attr):
            data = json.loads(attr)
            return (data.get('ignition', None),(data.get('hours', None)))
    columns = ['ignition', 'hours']
    df[columns] = df['ATTRIBUTES'].apply(extract_attributes).tolist()
    df.drop('ATTRIBUTES', axis=1, inplace=True)
    next_df = df.shift(-1) 
    time_diff = next_df['time'] - df['time']
    weh_msk  = (df['ignition'] == True)
    try:
        total_time_working_engine_hours = time_diff[weh_msk].dt.total_seconds().sum()
    except:
        total_time_working_engine_hours = 0
    if (df['hours'].iloc[-1] - df['hours'].iloc[0])/36000000 - (total_time_working_engine_hours/3600) in (1,-1):
        return total_time_working_engine_hours
    # return ((df['hours'].iloc[-1] - df['hours'].iloc[0])/3600000).__round__(2)
    return total_time_working_engine_hours/3600


def get_distance(df):
    df = df.copy()
    next_df = df.shift(-1)
    df["prev_LAT"] = df["LAT"].shift(1)
    df["prev_LON"] = df["LON"].shift(1)
    lat1 = df['LAT'].to_numpy()
    lon1 = df['LON'].to_numpy()
    lat2 = next_df['LAT'].to_numpy()
    lon2 = next_df['LON'].to_numpy()            
    distances = haversine(lat1, lon1, lat2, lon2)
    return distances.sum()

def calculate_group_distances(df):
    df['group'] = (df['EV'] != df['EV'].shift()).cumsum()

    df["LAT2"] = df["LAT"].shift(-1)
    df["LON2"] = df["LON"].shift(-1)
    df["EV2"] = df["EV"].shift(-1)

    is_last_in_group = df['group'] != df['group'].shift(-1)
    df = df[~is_last_in_group].copy()

    # إعادة shift بعد الفلترة
    df["LAT2"] = df["LAT"].shift(-1)
    df["LON2"] = df["LON"].shift(-1)
    df["EV2"] = df["EV"].shift(-1)

    valid_mask = (
        (df['EV'] == 1) &
        (df['EV2'] == 1) &
        df['LAT'].notna() &
        df['LON'].notna() &
        df['LAT2'].notna() &
        df['LON2'].notna()
    )

    lat1 = df.loc[valid_mask, 'LAT'].to_numpy()
    lon1 = df.loc[valid_mask, 'LON'].to_numpy()
    lat2 = df.loc[valid_mask, 'LAT2'].to_numpy()
    lon2 = df.loc[valid_mask, 'LON2'].to_numpy()

    distances = haversine(lat1, lon1, lat2, lon2)
    distances[distances >= 10] = 0

    if valid_mask.sum() != len(distances):
        raise ValueError(f"Mismatch: valid_mask has {valid_mask.sum()} but distances has {len(distances)}")

    df.loc[valid_mask, 'DIST'] = distances
    df['DIST'] = df['DIST'].fillna(0)

    return df.groupby('group')['DIST'].sum()





def get_distance_from_db(cbox: str, start_day: str, end_day: str):
    print("start day", start_day)
    print("end day", end_day)
    query = """
        SELECT SUM(daily_distance)
        FROM veh_metrics
        WHERE cbox = ? AND the_day >= ? AND the_day <= ?
    """
    cursor = connection.cursor()
    cursor.execute(query, (cbox, start_day, end_day))
    result = cursor.fetchone()
    return result[0] if result[0] is not None else 0

def get_journal_metrics(df: pd.DataFrame, start_day, end_day, distanceMethodeCalcType, threshold=60) -> tuple:
    print("distanceMethodeCalcType  ", distanceMethodeCalcType)
    next_df = df.shift(-1)
    df["prev_LAT"] = df["LAT"].shift(1)
    df["prev_LON"] = df["LON"].shift(1)    
    time_diff = next_df['time'] - df['time']
    df['time_diff'] = df['time'].diff().dt.total_seconds()
    df['is_stop'] = df['EV'].isin([2, 4])
    stop_change = (df['is_stop'] != df['is_stop'].shift()) | (~df['is_stop'])
    df['stop_group'] = stop_change.cumsum()
    mask_travel_hours = (
        (df['EV'] == 1) & 
        (df['SPD'] > 0) & 
        (next_df['EV'] == 1) &  
        (next_df['SPD'] > 0)  
    )
    mask_working_engine_hours = ( (df['EV'] == 1) | (df['EV'] == 4) ) 
    mask_en_arreit = ((df['EV'] == 2) | (df['EV'] == 4) )
    distance = 0

    if distanceMethodeCalcType == 1:
        distance = get_distance_from_db(df['CBOX'].iloc[0], start_day + timedelta(days=1) , end_day)
    else:
        oldDistance = get_distance_from_db(df['CBOX'].iloc[0], start_day + timedelta(days=1), end_day )
        df['date'] = df['time'].dt.date
        mask_distance = (df['SPD'] > 0) & (next_df['SPD'] > 0) & (df['date'] == end_day.date())
        print(mask_distance.tail())
        lat1 = df['LAT'][mask_distance].to_numpy()
        lon1 = df['LON'][mask_distance].to_numpy()
        lat2 = next_df['LAT'][mask_distance].to_numpy()
        lon2 = next_df['LON'][mask_distance].to_numpy()            
        distances = haversine(lat1, lon1, lat2, lon2)
        distance  = distances.sum() + oldDistance

    stop_groups = df[df['is_stop']].groupby('stop_group')['time_diff'].sum()
    try:
        total_time_seconds_travel_hours = time_diff[mask_travel_hours].dt.total_seconds().sum()
    except:
        total_time_seconds_travel_hours = 0
    try:    
        total_time_working_engine_hours = time_diff[mask_working_engine_hours].dt.total_seconds().sum()
    except:
        total_time_working_engine_hours = 0
    try:
        total_arreit = time_diff[mask_en_arreit].dt.total_seconds().sum()
    except:
        total_arreit = 0
    try:
        nb_arret = (stop_groups >= threshold).sum()
    except:
        nb_arret = 0
    
    total_time_en_arreit =total_arreit


    return   {
                'DailyDistance': distance,
                'travelHours':total_time_seconds_travel_hours / 3600,
                "stopCount":nb_arret,
                "stopDuration":total_time_en_arreit/3600,
                "engineWorkingHours":total_time_working_engine_hours / 3600
            }
   



