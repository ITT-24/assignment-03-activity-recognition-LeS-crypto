# this program recognizes activities

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, scale, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import svm
from pathlib import Path
import collections


ACT_IDS = {'running': 0,'rowing': 1, 'lifting': 2, 'jumpingjacks': 3}
HEADER = ['activity', 'timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'] 
df = pd.DataFrame(columns=HEADER) # create an empty df
test_size = 0.2
data_dir = "data/"

COLS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'] # + timestamp -> acc_score = 1
# NOTE: use timestamp-diff btw lows and peaks

class Recognizer:
    # init empty df's
    df = pd.DataFrame(columns=HEADER) 
    train = pd.DataFrame(columns=HEADER)
    test = pd.DataFrame(columns=HEADER)
    classifier = None

    def read_data():
        # read, preprocess, feature extract and then split all activities separtely , then append to 1
        df_act = pd.DataFrame(columns=HEADER)

        folder = Path(data_dir)
        for key, value in ACT_IDS.items():
            for filepath in folder.glob(f"*{key}-*.csv"):
                if df_act.empty:
                    df_act = pd.read_csv(filepath, header=0)
                    names = str(filepath).split("-")
                    df_act['activity'] = ACT_IDS[names[1]]
                else:
                    df_file =  pd.read_csv(filepath, header=0)
                    names = str(filepath).split("-")
                    df_file['activity'] = ACT_IDS[names[1]]
                    df_act = pd.concat([df_act, df_file], ignore_index=True)
                #TODO
                # print("test", df_act)
            if not df_act.empty:
                # print(df_act)
                df_act = Recognizer.preprocess_data(df_act)
                df_act = Recognizer.feature_extraction(df_act)
                Recognizer.split_data(df_act)
                df_act = pd.DataFrame(columns=HEADER)

                # concat ??
                # if Recognizer.df.empty:
                #     Recognizer.df = df_act
                # else:
                #     Recognizer.df = pd.concat([Recognizer.df, df_act], ignore_index=True)

    def preprocess_data(df:pd.DataFrame):
        print("Preprocessing the data")
        df = df.copy()
        df = df.dropna()

        # # set timestamp as index
        # df.set_index('timestamp', inplace=True)
        # df.sort_index(inplace = True)

        return df
    

    def feature_extraction(df:pd.DataFrame):
        # see: https://machinelearningmastery.com/moving-average-smoothing-for-time-series-forecasting-python/
        # get the moving average with a window size of 2s
        df = df.copy()
        cols = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']


        # normalize time
        start_time = df.iloc[0]['timestamp']
        # print("start", start_time)
        df[['timestamp']] = df[['timestamp']].transform(lambda x: x-start_time) # works
        # print(df.tail)

        # get frequency of 1 excercise cycle ~~ 2s = 200ms
        RATE = 10 # /df.iloc[-1]['timestamp'] # 1 sample every 10ms
        CYCLE = 20 # 2s = 20 samples

        # TRANSFORM SENSOR DATA TO FREQUENCY DOMAIN
        data = df[COLS].to_numpy()
        # print("data", np.shape(data))
        fft_data = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(fft_data))
        # print("freqs", freqs, np.shape(freqs))

        df["frequency"] = pd.Series(freqs)
        df = df.dropna()
        print("FREQUENCIES:\n", df.head())

        # STANDARDIZATION
        scaled_samples = scale(df[cols])
        df_mean = df.copy()
        df_mean[cols] = scaled_samples

        print("STAND:\n", df_mean.head())

        # NORMALIZATION
        scaler = MinMaxScaler()
        scaler.fit(X=df_mean[cols])
        scaled_samples = scaler.transform(X=df_mean[cols])
        df_normalized = df_mean.copy()
        df_normalized[cols] = scaled_samples

        print("NORM:\n", df_normalized.head())

        # train model to predict label for that cycle
        # create lagged feature ?

        return df_normalized


    def split_data(df:pd.DataFrame):
        split = round(0.80 * len(df))
        train = df.iloc[:split]
        test = df.iloc[split:]
        if Recognizer.train.empty and Recognizer.test.empty:
            Recognizer.train = train
            Recognizer.test = test
        else:
            Recognizer.train = pd.concat([Recognizer.train, train], ignore_index=True)
            Recognizer.test = pd.concat([Recognizer.test, test], ignore_index=True)
        print(f"train: {Recognizer.train.shape} - // - test: {Recognizer.test.shape}")


    def train_model():
        train = Recognizer.train

        # CLASSIFIER
        # classifier = svm.SVC(kernel='linear') # score = 0.81
        classifier = svm.SVC(kernel='rbf') # score = 0.98

        cols = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'frequency']

        classifier.fit(train[cols], train['activity']) 
        class_score = classifier.score(train[cols], train['activity'])

        print("CLASS:", classifier, class_score)

        Recognizer.classifier = classifier
        

    def evaluate_training():
        print("evaluating training")
        cols = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'frequency']

        classifier = Recognizer.classifier
        predicted_act = classifier.predict(Recognizer.test[cols])
        print("PREDICTION", predicted_act) # array with labels

        acc_score = accuracy_score(predicted_act, Recognizer.test[["activity"]])
        print("ACC SCORE:", acc_score)
        print("CLASSIFICATION REPORT:\n", classification_report(Recognizer.test[["activity"]], predicted_act))
        print("CONFUSION MATRIX:\n", confusion_matrix(Recognizer.test[['activity']], predicted_act, labels=[0, 1, 2, 3]))


    def predict_activity(sample:pd.DataFrame):
        classifier = Recognizer.classifier
        prediction = classifier.predict(sample)
        # print("PREDICT: ", ACT_IDS['jumpingjacks'], prediction) 
        return prediction



def train_model():
    Recognizer.read_data()
    Recognizer.train_model()
    Recognizer.evaluate_training()
    return True

# IDEA: get array of 10/20 length to predict on
def use_model(input_data:list[list]=None) -> str:
    """Uses the trained model to predict the exercize being performed. 
    Returns the activity as a string."""

    cols = COLS
    cols.append('timestamp')

    if input_data != None:
        # sample = np.reshape(input_data, (1, -1))
        sample = input_data
        # print("sample", sample)
        # sample = input_data
        sample = pd.DataFrame(sample, columns=['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
        print("sample\n", sample)

        # PREPROCESS DATA
        sample = Recognizer.preprocess_data(sample)
        sample = Recognizer.feature_extraction(sample)

        # check if data same length
        # preprocess 
        # Model should predict activities based on sensor data from DIPPID
        # continuosly run & return prediction on user input
        # parse input data to same "syntax"
        # return key, value pair (i.e. {"running": 0})
        prediction = Recognizer.predict_activity(sample)
        print("PREDICT: ", prediction)

        # TODO: if majority is one sample -> return that
        # unique, counts = np.unique(prediction, return_counts=True)
        # predicted_acts = dict(zip(unique, counts))
        # print(predicted_acts)
        # major_act = predicted_acts[0]

        # see: https://stackoverflow.com/a/28663910
        predicted_acts = collections.Counter(prediction) # majority = first idx
        # print(predicted_acts)
        major_act = list(predicted_acts.keys())[0]
        print("Predicted Activity ==", major_act)
        
        for key, value in ACT_IDS.items():
            if value == major_act:
                print(f"Predicted Activity == {key}")
        #         return key # ?
    else:
        print("No data recieved")


if __name__ == "__main__":
    # for testing
    train_model()
    # if train_model():
    # use_model([1715411336750, 1.74,2.62,-0.97,373.54,-129.27,-431.76]) # juming jacks (without)

    # IDEA: collect data for a little bit, then predict the activity
    test_array = [[1715411324450,2.76,0.91,0.61,255.43000000000004,-282.1,-369.81],
                  [1715411324460,2.76,0.91,0.61,255.43,-282.1,-369.81],
                  [1715411324470,2.76,0.9100000000000001,0.61,255.42999999999998,-282.1,-369.81],
                  [1715411324480,2.977540106951871,0.6751336898395722,0.9045454545454545,247.98417112299467,-277.9602673796792,-362.7388770053476],
                  [1715411324490,5.02,-1.53,3.67,56.519999999999996,-171.51,-180.91],
                  [1715411324500,5.02,-1.53,3.67,56.52,-171.51,-180.91],
                  [1715411324510,5.02,-1.53,3.67,56.52,-171.51,-180.91],
                  [1715411324520,5.000208333333333,-1.7615625000000001,3.8441666666666663,55.80411458333334,-170.24223958333332,-179.57421875],
                  [1715411324530,4.92,-2.7,4.55,-80.92999999999999,71.9,75.56000000000002],
                  [1715411324540,4.92,-2.7000000000000006,4.55,-80.92999999999999,71.9,75.56],
                  [1715411324550,4.92,-2.7000000000000006,4.55,-80.93,71.9,75.56]]
    # should be jumping jacks i.e 3
    
    use_model(test_array)