# this program recognizes activities

import activity_recognizer as activity
import pyglet
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import svm
from pathlib import Path


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
        """Read all .csv data and put them into a single dataframe."""

        # see: https://docs.python.org/3/library/pathlib.html
        folder = Path(data_dir)
        for filepath in folder.glob("*.csv"):
            if Recognizer.df.empty:
                Recognizer.df = pd.read_csv(filepath, header=0)
                names = str(filepath).split("-")
                Recognizer.df['activity'] = ACT_IDS[names[1]]
            else:
                file_df = pd.read_csv(filepath, header=0)
                names = str(filepath).split("-")
                file_df['activity'] = ACT_IDS[names[1]]
                Recognizer.df = pd.concat([Recognizer.df, file_df], ignore_index=True)
            # NOTE: otherwise throws FutureWarning

        # make all the same length?
        print(Recognizer.df.head(), "\n", Recognizer.df.shape)

    # def convert_datetime(df):
    #     """Convert the pd.datetime object to a float"""
    #     # kinda: https://stackoverflow.com/a/74490344 and https://stackoverflow.com/a/24590666
    #     df["timestamp"] = pd.to_datetime(df["timestamp"]).values.astype('float64')

    def preprocess_data():
        print("Preprocessing the data")
        df = Recognizer.df.copy()

        df = df.dropna()

        return df
        # NOTE: https://scikit-learn.org/stable/modules/impute.html#impute

    def feature_extraction():
        df = Recognizer.preprocess_data()


        # STANDARDIZATION
        scaled_samples = scale(df[COLS])
        df_mean = df.copy()
        df_mean[COLS] = scaled_samples

        print("STAND:\n", df_mean.head())

        # NORMALIZATION
        scaler = MinMaxScaler()
        scaler.fit(X=df_mean[COLS])
        scaled_samples = scaler.transform(X=df_mean[COLS])
        df_normalized = df_mean.copy()
        df_normalized[COLS] = scaled_samples

        print("NORM:\n", df_normalized.head())

        # TODO: fft
        # apply fft to all COLS -> per activity
        # df_fft = df_normalized[COLS].apply(np.fft.fft)
        # # extract peaks
        # df_fft = df_fft[COLS].apply(np.abs)
        # print("FFT:\n", df_fft.head(), df_fft.shape)

        # (e.g. filtering, normalization, transformation into frequency domain (fft), ...)
        # NOTE: right now on all values -> do only for specific columns?? acc and gyro (and timestamp)


        # split 80/20
        Recognizer.train, Recognizer.test = train_test_split(df_normalized, test_size=test_size)
        print(f"train: {Recognizer.train.shape} - // - test: {Recognizer.test.shape}")

    def train_data():
        # train ml classifier on train
        train = Recognizer.train

        # CLASSIFIER
        # classifier = svm.SVC(kernel='linear') # score = 0.81
        classifier = svm.SVC(kernel='rbf') # score = 0.98

        classifier.fit(train[COLS], train['activity']) 
        class_score = classifier.score(train[COLS], train['activity'])

        print("CLASS:", classifier, class_score)

        Recognizer.classifier = classifier

    def evaluate_training():
        classifier = Recognizer.classifier
        predicted_act = classifier.predict(Recognizer.test[COLS])
        # print("PREDICTION", predicted_act) # array with labels
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
    Recognizer.feature_extraction()
    Recognizer.train_data()
    Recognizer.evaluate_training()
    return True

def use_model(input_data:list=None):
    if input_data != None:
        sample = np.reshape(input_data, (1, -1))
        sample = pd.DataFrame(sample, columns=COLS)

        # check if data same length
        # preprocess 
        # Model should predict activities based on sensor data from DIPPID
        # continuosly run & return prediction on user input
        # parse input data to same "syntax"
        # return key, value pair (i.e. {"running": 0})
        prediction = Recognizer.predict_activity(sample)
        print("PREDICT: ", prediction)
        for key, value in ACT_IDS.items():
            if value == prediction:
                print(f"{prediction} == {key}")
                return {key: value} # ?
    else:
        print("No data recieved")
    pass


if __name__ == "__main__":
    if train_model():
        use_model([1.74,2.62,-0.97,373.54,-129.27,-431.76])