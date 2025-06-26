from influxdb import InfluxDBClient
import pandas as pd
import sys
import fdb
import time
from datetime import datetime, timedelta
from details import allMetricsCalcResult
# Influx DB for time series data
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
    def __init__(self, query) -> None:
        self.query = query
        self.result = None
        self.points = None
        self.df = None
        self.is_empty = False

    def setResult(self):
        self.result = client.query(query=self.query)

    def setPoints(self):
        self.points = self.result.get_points()

    def setDf(self):
        self.df = pd.DataFrame(self.points)

    def checkIsEmpty(self):
        self.is_empty = len(self.df) <= 1

# Firebird database connection details
host = '10.10.1.26'
# host='127.0.0.1'
user = 'sysdba'
password = 'bmw'
port = 3050
con_str = f"{host}/{port}:/opt/gps/backend/fc_param.fdb"
QUERRVEH = """
    INSERT INTO VEH_METRICS(
        THE_DAY, CVEH, CBOX, DAILY_DISTANCE, TRAVEL_HOURS, START_TIME, END_TIME, 
        DAILY_FUEL, IDLE_HOURS, REFILLS_NUMBER, LITERS_FILLED, MAX_TRAVEL_HOURS,
        AVERAGE_SPEED, 
        REGULATOR_USAGE_MINUTES, MAX_SPEED, AVERAGE_ECO_SCORE, NB_HARD_BRAKING,
        NB_HARD_ACCELERATION, NB_FATIGUE_DRIVING,FUEL_COST,STOP_COUNT,STOP_DURATION,ENGINE_WORKING_HOURS
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?,?,?)
           """

def create_connection():
    try:
        connection = fdb.connect(con_str, user=user, password=password)
        return connection
    except Exception as e:
        print(f"Error connecting to Firebird: {e}")
        sys.exit(1)


# def getDateInsert():  
#     connection = create_connection()  
#     cursor = connection.cursor()  
#     cursor.execute("SELECT MIN(THE_DAY) FROM veh_metrics")  
#     res = cursor.fetchone()
#     cursor.close()
#     connection.close()  
#     return res[0]


def fetch_vehicle_data():
    query = "select CVEH, CBOX from Veh"
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    connection.close()
    return [{"CVEH": row[0], "CBOX": row[1]} for row in rows]
cbox_unidev_list = fetch_vehicle_data()
    
def insert_data_batch(generator, connection, batch_size=10):
    cursor = connection.cursor()
    batch = []
    for data in generator:
        batch.append(data)
        if len(batch) == batch_size:
            try:
                cursor.executemany(QUERRVEH, batch)
                connection.commit()
            except Exception as e:
                print(f"Error inserting batch: {e}")
            batch = []
    if batch:
        try:
            cursor.executemany(QUERRVEH, batch)
            connection.commit()
        except Exception as e:
            print(f"Error inserting final batch: {e}")
    
def traiter_and_insert(day, cbox_unidev_list):
    connection = create_connection()    
    insert_data_batch(traiter(day, cbox_unidev_list) , connection)
    connection.close()

def traiter(day, cbox_unidev_list):
    prev_day = day - timedelta(days=1)
    for cbox in cbox_unidev_list:
        condition = f"CBOX = '{cbox['CBOX']}'"       
        veh = Journal(f"SELECT * FROM journal WHERE time >= '{prev_day}' AND time <= '{day}' AND {condition}")
        veh.setResult()
        veh.setPoints()
        veh.setDf()
        veh.checkIsEmpty()
        if not veh.is_empty:
            data = allMetricsCalcResult(veh.df)
            yield(
                data['date'],
                cbox["CVEH"],
                cbox['CBOX'],
                data['DailyDistance'],
                data['travelHours'],
                data['startTime'],
                data['endTime'],
                data['dailyFuel'],
                data['IdleHours'],
                data['refillsNumber'],
                data['litersFilled'],
                data['maxTravelHours'], 
                int(data['averageSpeed']),
                str(data['regulatorUsageMinutes']),
                int(data['maxSpeed']),
                data['score'],
                int(data['NombreFatime']),
                int(data['nbHardBraking']),
                int(data['nbHardAcceleration']),
                data['fuel_cost'],
                data['stop_count'],
                data['stop_duration'],
                data['engine_working_hours'])

today = datetime.now().date() 
print("start")
start_time = time.time()
traiter_and_insert(today, cbox_unidev_list)
print(f"Time taken for  ({today-timedelta(days=1)}) is {time.time() - start_time:.2f} seconds")
