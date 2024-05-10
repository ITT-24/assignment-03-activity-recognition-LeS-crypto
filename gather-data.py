from DIPPID import SensorUDP
import pandas as pd
# from time import sleep, time
import time
import datetime
import sys
import numpy as np
# this program gathers sensor data

"""
- [ ] (2P) data is logged correctly
- [ ] (1P) log files are named and structured appropriately
- [ ] (1P) logging can be started with the DIPPID device
- [ ] (1P) enough data sets captured
"""

# TODO
    # press button_1 on dippid = start capturing
    # File Name: your_name-activity-number.csv (e.g., susi-running-1.csv)
    # Columns: id,timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
    # stop automatically after enough data is captured (1.000 rows?)
    # record 5 sets of running, rowing, lifting, jumpingjacks
    # Resample to 100 Hz = 100 datapoints per second / points 10ms apart
    # upload data to GRIPS repo

"""
frequency of data https://datascience.stackexchange.com/a/77300
    - is the difference between consecutive time-stamps
    - 1 Hz = once per second, i.e. 10ms apart = 100 times per second = 100 Hz
downsampling https://stackoverflow.com/a/57845046
    - pandas -> resample
sample at 100 Hz https://stackoverflow.com/a/57151232
"""

PORT = 5700
sensor = SensorUDP(PORT)
CSV_HEADER = ['id', 'timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'] 
ACTIVITIES = {0: 'running', 1 : 'rowing', 2: 'lifting', 3: 'jumpingjacks'}
SAMPLE_LIMIT = 100000  
INTERATION_LIMIT = 5 # HACK for test
# DATA_SHAPE = (8, 8)

class Activity:

    def __init__(self, name) -> None:
        self.name = name
        self.iteration = 0
        # self.data = np.zeros(DATA_SHAPE)
        self.data = []

    @classmethod
    def select_acitivity(cls):
        print("Record data for which activity?")
        for i in range(0, len(ACTIVITIES)):
            print(f"ID: {i} - {ACTIVITIES[i]}")
        activity_id = int(input()) 
        print("will record", activity_id, ACTIVITIES[activity_id])

        act = cls.init_activity(activity_id)
        return act

    @classmethod
    def init_activity(cls, id:int):
        """Create the chosen child activity"""
        match id:
            case 0:
                return Running()
            case 1:
                return Rowing()
            case 2:
                return Lifting()
            case 3: 
                return Jumping_Jacks()
            case _:
                print("ERROR")
                return None

    def record_data_row(self, row:list):
        # self.data[self.iteration] = row
        self.data.append(row)
        # print("i", self.iteration, " -> ", row, "=", len(self.data))

    def iterate_activity(self):
        self.save_as_csv()
        self.iteration += 1
        self.data = [] # reset

    def save_as_csv(self):
        # TODO: resample to 100 HZ

        filepath = f"data/leonie-{self.name}-{self.iteration}.csv"
        df = pd.DataFrame(self.data, columns=CSV_HEADER)

        df['time'] = pd.to_datetime(df['timestamp']) # create for resampling -> used as "index"
        df = df.resample('10ms', on='time').mean()

        # print(df.head())

        df.to_csv(filepath, index=False, header=CSV_HEADER)
        # print("create", filepath)
    
    # def resample_data(df:pd.DataFrame)-> pd.DataFrame:
    #     df.resample('10L')
    #     pass


# ----- CHILD ACTIVITIES ---- #
class Running(Activity):
    def __init__(self) -> None:
        self.id = 0
        super().__init__(name="running")

class Rowing(Activity):
    def __init__(self) -> None:
        self.id = 1
        super().__init__(name="rowing")

class Lifting(Activity):
    def __init__(self) -> None:
        self.id = 2
        super().__init__(name="lifting")

class Jumping_Jacks(Activity):
    def __init__(self) -> None:
        self.id = 3
        super().__init__(name="jumpingjacks")



if __name__ == '__main__':

    # dt = pd.DatetimeIndex(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])
    # print(pd.to_datetime(pd.Timestamp(time.time())))
    # print(datetime.datetime.now())
    # print(pd.to_datetime(datetime.datetime.now()))

    act = Activity.select_acitivity()

    has_acc  = sensor.has_capability('accelerometer')
    has_gyro = sensor.has_capability('gyroscope')
    has_butt = sensor.has_capability('button_1')
    print("DIPPID devices is sending data:", has_butt)

    if act == None and has_butt == False:
        print("Something went wrong D:")
    else:
        print("press button_1 to start recording")

        # while(True):
        # start = sensor.get_value('button_1')

        start = False
        while start == False:
            start = sensor.get_value('button_1')

        if start: #  and act.iteration < INTERATION_LIMIT:
            print("start")
            time.sleep(2) # skip the pause in action 
            while act.iteration < INTERATION_LIMIT: # in range(0, INTERATION_LIMIT):
                if len(act.data) < SAMPLE_LIMIT:
                    # Record all relevant data
                    current_id = act.id
                    t = datetime.datetime.now()
                    acc  = sensor.get_value('accelerometer')
                    gyro = sensor.get_value('gyroscope')

                    row = [current_id, t, acc['x'], acc['y'], acc['z'], gyro['x'], gyro['y'], gyro['z']]

                    act.record_data_row(row)

                    print("+", len(act.data)) # HACK -> doesn't finish that fast
                else: 
                    time.sleep(1)
                    print("start next iteration")
                    act.iterate_activity()

        print("finish")
