# this program recognizes activities

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.feature_selection import VarianceThreshold
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
MODEL_COLS = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'frequency']
MODEL_COLS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'frequency']

# NOTE: use timestamp-diff btw lows and peaks

# NOTE: how to work around idle-sensor data??


class Recognizer:
    # init empty df's
    df    : pd.DataFrame = pd.DataFrame(columns=HEADER) 
    train : pd.DataFrame = pd.DataFrame(columns=HEADER)
    test  : pd.DataFrame = pd.DataFrame(columns=HEADER)
    classifier : svm.SVC

    @classmethod
    def read_data(cls):
        # read, preprocess, feature extract and then split all activities separtely , then append to 1 df
        df_act = pd.DataFrame(columns=HEADER)

        folder = Path(data_dir)
        for key, value in ACT_IDS.items():
            for filepath in folder.glob(f"*{key}-*.csv"): #(f"*{key}-*.csv"):
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
                df_act = cls.preprocess_data(df_act)
                df_act = cls.feature_extraction(df_act)
                # Recognizer.split_data(df_act)
                cls.split_data_randomly(df_act)
                df_act = pd.DataFrame(columns=HEADER)

                # concat ??
                # if Recognizer.df.empty:
                #     Recognizer.df = df_act
                # else:
                #     Recognizer.df = pd.concat([Recognizer.df, df_act], ignore_index=True)
        # Recognizer.split_data_randomly(df_act)

    @classmethod
    def preprocess_data(cls, df:pd.DataFrame) -> pd.DataFrame:
        print("Preprocessing the data")
        df = df.copy()
        df = df.drop(['timestamp'], axis=1)
        df = df.dropna()

        # # set timestamp as index
        # df.set_index('timestamp', inplace=True)
        # df.sort_index(inplace = True)

        return df
    

    @classmethod
    def feature_extraction(cls, df:pd.DataFrame) -> pd.DataFrame:
        # see: https://machinelearningmastery.com/moving-average-smoothing-for-time-series-forecasting-python/
        # get the moving average with a window size of 2s
        df = df.copy()
        # cols = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        cols = MODEL_COLS

        # normalize time
        # start_time = df.iloc[0]['timestamp']
        # df[['timestamp']] = df[['timestamp']].transform(lambda x: x-start_time)
        # weglassen

        # TRESHOLD SENSOR DATA -> learn only big changes
        # selector = VarianceThreshold(threshold=df[COLS].mean())
        # selector.fit(df[COLS])
        # print("selector", selector)
        #df[COLS] = df[COLS].where(df[COLS] > df[COLS].mean())
        # print(df[COLS].min())
        # treshold = df[COLS].min()
        # treshold = pd.Series(treshold.to_numpy())
        # print(treshold.to_numpy())
        # df[COLS] = df[COLS].clip(lower=treshold, axis=0)
        # df = df.dropna()
        # print("THRESHOLD", df[COLS])
        # df[COLS] = df[COLS].where(df[COLS] > df[COLS].min())  

        # Give weight to big diff changes       



        # TRANSFORM SENSOR DATA TO FREQUENCY DOMAIN
        data = df[COLS].to_numpy()
        fft_data = np.fft.rfft(data)
        freqs = np.fft.fftfreq(len(fft_data)) # get all frequencies
        # then insert back into df
        df["frequency"] = pd.Series(freqs)
        df = df.dropna()
        # print("FREQUENCIES:\n", df)

        # TODO: DETECT SPIKES/PEAKS IN (FREQ) DATA (?? give more weight)
        # only use data to train, that has significant movement
        # https://gist.github.com/w121211/fbd35d1a8776402ac9fe24654ca8044f
        # https://stackoverflow.com/questions/60794064/step-spike-detection-breakdown-for-pandas-dataframe-graph
        
        # TRESHOLD FREQUENCY
        # print(df["frequency"].mean())
        # df["frequency"] = df["frequency"].where(df['frequency'].abs() > np.abs(df['frequency'].mean()))
        # df = df.fillna(0)
        # print("FREQ", df["frequency"])
        # DOES NOT CHANGE ANYTHING

        # Centered Moving Average
        # rolling = df["frequency"].rolling(window=5)
        # print("rolling", rolling)
        # rolling_mean = rolling.mean()
        # rolling_mean = rolling_mean.fillna(0)
        # print("rolling_mean", rolling_mean)
        # df["rolling"] = rolling_mean



        # DIFFERENTIATION -> get difference in frequency
        # df["difference"] = df["frequency"].diff()
        # df = df.fillna(0) # replace NaN in first entry
        # print("DIFFERENCE\n", df["difference"].mean(), df['difference'])
        # df["difference"] = np.abs(df["difference"]).where(np.abs(df['difference']) == np.abs( df["difference"].mean()) )
        # df = df.fillna(0)
        # ??

        # WEIGHTS -> more weights, where difference is bigger -> use diff
        # print("mean", df["frequency"].mean())
        # df["weight"] = np.abs(df["frequency"]).where(np.abs(df['frequency']) > np.abs( df["frequency"].mean()) )


        # STANDARDIZATION
        scaled_samples = scale(df[cols])
        df_mean = df.copy()
        df_mean[cols] = scaled_samples

        # print("STAND:\n", df_mean.head())

        # NORMALIZATION
        scaler = MinMaxScaler()
        scaler.fit(X=df_mean[cols])
        scaled_samples = scaler.transform(X=df_mean[cols])
        df_normalized = df_mean.copy()
        df_normalized[cols] = scaled_samples

        # print("NORM:\n", df_normalized.head())


        # train model to predict label for that cycle
        # create lagged feature ?

        return df_normalized


    @classmethod
    def split_data(cls, df:pd.DataFrame):
        """Split by time
        see: # https://medium.com/@mouse3mic3/a-practical-guide-on-scikit-learn-for-time-series-forecasting-bbd15b611a5d
        """

        split = round(0.80 * len(df))
        train = df.iloc[:split]
        test = df.iloc[split:]
        if cls.train.empty and cls.test.empty:
            cls.train = train
            cls.test = test
        else:
            cls.train = pd.concat([cls.train, train], ignore_index=True)
            cls.test = pd.concat([cls.test, test], ignore_index=True)
        # print(f"train: {Recognizer.train.shape} - // - test: {Recognizer.test.shape}")

    @classmethod
    def split_data_randomly(cls, df:pd.DataFrame): # test!!
        """Split randomly"""
        if cls.train.empty and cls.test.empty:
            cls.train, cls.test = train_test_split(df, test_size=test_size)
        else:
            train, test = train_test_split(df, test_size=test_size)
            cls.train = pd.concat([cls.train, train], ignore_index=True)
            cls.test = pd.concat([cls.test, test], ignore_index=True)


    @classmethod
    def train_model(cls):
        train = cls.train
        print("TRAIN:\n", train)
        cols = MODEL_COLS
        # cols = TEST_COLS

        # CLASSIFIER
        # classifier = svm.SVC(kernel='linear') # score = 0.62
        classifier = svm.SVC(kernel='rbf') # score = 0.933
        # classifier = svm.SVC(kernel='poly') # score = 0.935
        # classifier = svm.SVC(kernel='sigmoid') # score = 0.109
        # classifier = svm.LinearSVC(dual='auto') # score = 0.639

        # "SVM berücksichtigt Timestamp nicht"


        classifier.fit(train[cols], train['activity']) 
        # classifier.fit(train[cols], train['activity'], sample_weight=train['weight']) 
        class_score = classifier.score(train[cols], train['activity'])

        print("CLASS:", classifier, class_score)

        cls.classifier = classifier
        

    @classmethod
    def evaluate_training(cls):
        print("evaluating training")
        cols = MODEL_COLS
        # cols = TEST_COLS

        classifier = cls.classifier
        predicted_act = classifier.predict(cls.test[cols])
        # print("PREDICTION", predicted_act) # array with labels

        acc_score = accuracy_score(predicted_act, cls.test[["activity"]])
        print("ACC SCORE:", acc_score)
        print("CLASSIFICATION REPORT:\n", classification_report(cls.test[["activity"]], predicted_act))
        print("CONFUSION MATRIX:\n", confusion_matrix(cls.test[['activity']], predicted_act, labels=[0, 1, 2, 3]))


    @classmethod
    def predict_activity(cls, sample:pd.DataFrame):
        classifier = cls.classifier
        prediction = classifier.predict(sample)
        # print("PREDICT: ", ACT_IDS['jumpingjacks'], prediction) 
        return prediction



def train_model():
    Recognizer.read_data()
    Recognizer.train_model()
    Recognizer.evaluate_training()
    return True

# IDEA: get array of 10/20 length to predict on
def use_model(input_data:list[list]) -> str:
    """Uses the trained model to predict the exercize being performed. 
    Returns the activity as a string."""

    result: str
    cols = COLS
    # cols.append('timestamp')

    if input_data != None:
        # sample = np.reshape(input_data, (1, -1))
        sample = input_data
        sample = pd.DataFrame(sample, columns=['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
        # PREPROCESS DATA
        # resample to 10ms -> like data
        # sample.set_index('timestamp', inplace=True)
        # df_resampled = sample.resample('10ms').mean()
        # df_resampled.reset_index(inplace=True)
        # df_resampled['timestamp'] = (df_resampled['timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')
        # df_resampled.index.name = 'id'

        sample = Recognizer.preprocess_data(sample)
        sample = Recognizer.feature_extraction(sample)

        print("sample\n", sample.head(), sample.shape)

        # PREDICTION
        prediction = Recognizer.predict_activity(sample)
        print("PREDICT: ", prediction)

        # see: https://stackoverflow.com/a/28663910
        prediction_count = collections.Counter(prediction) # majority = first idx
        print(prediction_count, list(prediction_count.keys()))
        act_id, _ = prediction_count.most_common(1)[0]
        # print("Predicted Activity ==", act_id)
        # {'running': 0,'rowing': 1, 'lifting': 2, 'jumpingjacks': 3}
        
        for key, value in ACT_IDS.items():
            if value == act_id:
                print(f"Predicted Activity == {key} ({act_id})")
        #         return key # ?
                result = key
    else:
        print("No data recieved")
        result = "..."
    return result


if __name__ == "__main__":
    # for testing
    train_model()
    # if train_model():
    # use_model([1715411336750, 1.74,2.62,-0.97,373.54,-129.27,-431.76]) # juming jacks (without)

    # IDEA: collect data for a little bit, then predict the activity
    # test_array = [[1715411324450,2.76,0.91,0.61,255.43000000000004,-282.1,-369.81],
    #               [1715411324460,2.76,0.91,0.61,255.43,-282.1,-369.81],
    #               [1715411324470,2.76,0.9100000000000001,0.61,255.42999999999998,-282.1,-369.81],
    #               [1715411324480,2.977540106951871,0.6751336898395722,0.9045454545454545,247.98417112299467,-277.9602673796792,-362.7388770053476],
    #               [1715411324490,5.02,-1.53,3.67,56.519999999999996,-171.51,-180.91],
    #               [1715411324500,5.02,-1.53,3.67,56.52,-171.51,-180.91],
    #               [1715411324510,5.02,-1.53,3.67,56.52,-171.51,-180.91],
    #               [1715411324520,5.000208333333333,-1.7615625000000001,3.8441666666666663,55.80411458333334,-170.24223958333332,-179.57421875],
    #               [1715411324530,4.92,-2.7,4.55,-80.92999999999999,71.9,75.56000000000002],
    #               [1715411324540,4.92,-2.7000000000000006,4.55,-80.92999999999999,71.9,75.56],
    #               [1715411324550,4.92,-2.7000000000000006,4.55,-80.93,71.9,75.56]]
    # should be jumping jacks i.e 3
    
    # use_model(test_array)