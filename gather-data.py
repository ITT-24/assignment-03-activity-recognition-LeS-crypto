from DIPPID import SensorUDP
import pandas as pd
import time
import datetime

PORT = 5700
sensor = SensorUDP(PORT)
CSV_HEADER = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'] 
ACTIVITIES = {0: 'running', 1 : 'rowing', 2: 'lifting', 3: 'jumpingjacks'}
SAMPLE_LIMIT = 200000  
INTERATION_LIMIT = 5 

class Activity:

    def __init__(self, name) -> None:
        self.name = name
        self.iteration = 0
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
        self.data.append(row)

    def iterate_activity(self):
        self.save_as_csv()
        self.iteration += 1
        self.data = [] # reset

    def save_as_csv(self):
        # AS: hard-coded name? :(
        filepath = f"data/leonie-{self.name}-{self.iteration}.csv"
        df = pd.DataFrame(self.data, columns=CSV_HEADER)

        # Resample to 100 Hz - see resample.py
        df.set_index('timestamp', inplace=True)
        df_resampled = df.resample('10ms').mean()
        df_resampled.reset_index(inplace=True)
        df_resampled['timestamp'] = (df_resampled['timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')
        df_resampled.index.name = 'id'

        df_resampled.to_csv(filepath, index=True, header=CSV_HEADER)


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

    act = Activity.select_acitivity()

    has_acc  = sensor.has_capability('accelerometer')
    has_gyro = sensor.has_capability('gyroscope')
    has_butt = sensor.has_capability('button_1')

    print("DIPPID devices is sending data:", has_butt)

    if act == None and has_butt == False:
        print("Something went wrong D:")
    else:
        print("press button_1 to start recording")

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

                    row = [t, acc['x'], acc['y'], acc['z'], gyro['x'], gyro['y'], gyro['z']]

                    act.record_data_row(row)

                    print("+", len(act.data)) # -> doesn't "record" that fast
                else: 
                    time.sleep(1)
                    print("start next iteration")
                    act.iterate_activity()

        print("finish")
