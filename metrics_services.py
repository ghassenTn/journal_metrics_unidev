from datetime import datetime
from influxdb import InfluxDBClient
import numpy as np
import pandas as pd
import sys
import json
from details import allMetricsCalcResult, get_working_engine_hours
from fastapi.encoders import jsonable_encoder
import constant as cs
INFLUXDB_HOST = cs.INFLUXDB_HOST
INFLUXDB_PORT = cs.INFLUXDB_PORT
INFLUXDB_DATABASE = cs.INFLUXDB_DATABASE

try:
    client = InfluxDBClient(host=INFLUXDB_HOST, port=INFLUXDB_PORT,database = INFLUXDB_DATABASE)
except Exception as e:
    print(f"Error connecting to InfluxDB: {e}")
    sys.exit(1)

class Journal:
    def __init__(self, query: str) -> None:
        self.query = query
        self._df = pd.DataFrame()
        self._is_loaded = False
        self._error = None
        self._load()

    def _load(self) -> None:
        """Execute the query and load the DataFrame."""
        try:
            result = client.query(self.query)
            points = result.get_points()
            self._df = pd.DataFrame(points)
            self._is_loaded = True
        except Exception as e:
            self._df = pd.DataFrame()
            self._is_loaded = False
            self._error = str(e)

    def reload(self) -> None:
        """Reload data from the same query."""
        self._load()

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def is_empty(self) -> bool:
        return self._df.empty

    @property
    def error(self) -> str:
        return self._error

    def __repr__(self) -> str:
        if self._is_loaded:
            return f"<Journal rows={len(self._df)}>"
        return f"<Journal ERROR: {self._error}>"


# Assuming Journal and allMetricsCalcResult, get_working_engine_hours are already imported or defined elsewhere.

def traiter(cbox, start_date, end_date):
    # start_date = datetime.fromisoformat(start_date)
    # end_date  = datetime.fromisoformat(end_date)
    name_mapping = {
        'date': 'THE_DAY',
        'DailyDistance': 'DAILY_DISTANCE',
        'dailyFuel': 'DAILY_FUEL',
        'litersFilled': 'LITERS_FILLED',
        'refillsNumber': 'REFILLS_NUMBER',
        'startTime': 'START_TIME',
        'endTime': 'END_TIME',
        'travelHours': 'TRAVEL_HOURS',
        'maxTravelHours': 'MAX_TRAVEL_HOURS',
        'NombreFatime': 'NB_FATIGUE_DRIVING',
        'nbHardBraking': 'NB_HARD_BRAKING',
        'nbHardAcceleration': 'NB_HARD_ACCELERATION',
        'IdleHours': 'IDLE_HOURS',
        'regulatorUsageMinutes': 'REGULATOR_USAGE_MINUTES',
        'maxSpeed': 'MAX_SPEED',
        'averageSpeed': 'AVERAGE_SPEED',
        'fuel_cost': 'FUEL_COST',
        'score': 'AVERAGE_ECO_SCORE',
        'stop_count': 'STOP_COUNT',
        'stop_duration': 'STOP_DURATION',
        'engine_working_hours': 'ENGINE_WORKING_HOURS'
    }

    condition = f"CBOX = '{cbox}'"
    veh = Journal(f"SELECT * FROM journal WHERE time >= '{start_date}' AND time <= '{end_date}' AND {condition}")
    print(veh.df.head())
    if not veh.is_empty:
        result = allMetricsCalcResult(veh.df)
        def clean_value(value):
            if isinstance(value, (np.integer, np.floating)):
                return value.item()
            elif isinstance(value, pd.Timestamp):
                return value.to_pydatetime()
            elif isinstance(value, (np.ndarray,)):
                return value.tolist()
            elif isinstance(value, datetime):
                return value
            else:
                return value
        mapped_result = {name_mapping.get(k, k): clean_value(v) for k, v in result.items()}
        return jsonable_encoder(mapped_result)
    else:
        d1 = pd.to_datetime(start_date)
        d2 = pd.to_datetime(end_date)
        mapped_result = {
            "THE_DAY": d1,
            "DAILY_DISTANCE": 0.0,
            "DAILY_FUEL": 0,
            "LITERS_FILLED": 0,
            "REFILLS_NUMBER": 0,
            "START_TIME": d1,
            "END_TIME": d2,
            "TRAVEL_HOURS": 0.0,
            "MAX_TRAVEL_HOURS": 0,
            "NB_FATIGUE_DRIVING": 0,
            "NB_HARD_BRAKING": 0,
            "NB_HARD_ACCELERATION": 0,
            "IDLE_HOURS": 0.0,
            "REGULATOR_USAGE_MINUTES": 0.0,
            "MAX_SPEED": 0.0,
            "AVERAGE_SPEED": 0.0,
            "FUEL_COST": 0,
            "AVERAGE_ECO_SCORE": 16.0,
            "STOP_COUNT": "1",
            "STOP_DURATION": float((d2 - d1).total_seconds() / 3600),
            "ENGINE_WORKING_HOURS": 0.0
        }

    return jsonable_encoder(mapped_result)


def traiter_ewh(cbox, start_date, end_date):
    name_mapping = {'engine_working_hours': 'ENGINE_WORKING_HOURS'}
    condition = f"CBOX = '{cbox.strip()}'"
    # print(f"SELECT * FROM journal WHERE time >= '{start_date.strip()}' AND time <= '{end_date.strip()}' AND {condition}")
    veh = Journal(f"SELECT * FROM journal WHERE time >= '{start_date.strip()}' AND time <= '{end_date.strip()}' AND {condition}")
    if not veh.is_empty:
        result = get_working_engine_hours(veh.df)
        mapped_result = {"ENGINE_WORKING_HOURS":result}
    else:
        mapped_result = {"ENGINE_WORKING_HOURS": 0.0}

    return jsonable_encoder(mapped_result)