# this program recognizes activities

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, scale, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import collections


ACT_IDS = {'running': 0,'rowing': 1, 'lifting': 2, 'jumpingjacks': 3}
HEADER = ['activity', 'timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'] 
# df = pd.DataFrame(columns=HEADER) # create an empty df
test_size = 0.2
data_dir = "data/"

COLS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'] # + timestamp -> acc_score = 1
MODEL_COLS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'frequency']

class Recognizer:
    # init empty df's
    df    : pd.DataFrame = pd.DataFrame(columns=HEADER) 
    train : pd.DataFrame = pd.DataFrame(columns=HEADER)
    test  : pd.DataFrame = pd.DataFrame(columns=HEADER)
    classifier : svm.SVC | RandomForestClassifier

    @classmethod
    def read_data(cls):
        """Read the different activity data sets and then process the data"""
        
        df_act = pd.DataFrame(columns=HEADER)
        folder = Path(data_dir)
        # see: https://docs.python.org/3/library/pathlib.html
        
        """ 
        NOTE: 
        Activities are processed individually before adding them to a complete dataset.
        This was initially done to try and use time-sensitive prediction (couldn't get that to work).
        But now used to better parse the different sensor readings to the frequency domain. 
        Tested regularly (a little) and this seemed to give better results
        """
        for key, _ in ACT_IDS.items():
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
            if not df_act.empty:

                # PROCESS THE DATA
                df_act = cls.preprocess_data(df_act)
                df_act = cls.feature_extraction(df_act)
                cls.split_data_randomly(df_act)
                df_act = pd.DataFrame(columns=HEADER)


    @classmethod
    def preprocess_data(cls, df:pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.drop(['timestamp'], axis=1) # "SVM berücksichtigt Timestamp nicht"
        df = df.dropna()
        return df

    @classmethod
    def feature_extraction(cls, df:pd.DataFrame) -> pd.DataFrame:
        """Transform the sensor data to frequencies, 
        then standardize and normalize and smooth"""

        df = df.copy()
        cols = MODEL_COLS   

        # TRANSFORM SENSOR DATA TO FREQUENCY DOMAIN
        data = df[COLS].to_numpy()
        fft_data = np.fft.rfft(data)
        freqs = np.fft.fftfreq(len(fft_data)) # get all frequencies
        # then insert back into df
        df["frequency"] = pd.Series(freqs)
        df = df.dropna()

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
        # print("NORM:\n", df_normalized.head(), df_normalized.shape)

        # FIND PEAKS and give give more weight
        # peaks, _ = signal.find_peaks(df_normalized['acc_x']) # indices of peaks in freqs
        # print("PEAKS-ACC_X", df_normalized['acc_x'].to_numpy()[peaks])
        # https://stackoverflow.com/a/60803014
        # ??


        # Smoothing the dataset
        rolling = df_normalized.rolling(window=20).mean()
        rolling = rolling.dropna()
        # print("ROLL:\n",rolling)
        # see https://towardsdatascience.com/sliding-windows-in-pandas-40b79edefa34
        # see https://pieriantraining.com/python-smoothing-data-a-comprehensive-guide/
        processed_df = rolling

        # processed_df = df_normalized
        return processed_df

    @classmethod
    def split_data_randomly(cls, df:pd.DataFrame):
        """Split the dataset randomly into 80/20 and then append to a full set"""
        if cls.train.empty and cls.test.empty:
            cls.train, cls.test = train_test_split(df, test_size=test_size)
        else:
            train, test = train_test_split(df, test_size=test_size)
            cls.train = pd.concat([cls.train, train], ignore_index=True)
            cls.test = pd.concat([cls.test, test], ignore_index=True)


    @classmethod
    def train_model(cls):
        cols = MODEL_COLS
        train = cls.train
        print("TRAIN:\n", train)

        # # CLASSIFIER
        # classifier = RandomForestClassifier() # test acc score = 0.979
        classifier = svm.SVC(kernel='rbf') # acc score = 0.973
        # classifier = svm.SVC(kernel='poly') # score = 0.965
        # classifier = svm.SVC(kernel='linear') # score = 0.731

        classifier.fit(train[cols], train['activity'])  
        class_score = classifier.score(train[cols], train['activity'])
        print("CLASS:", classifier, class_score)

        cls.classifier = classifier

    @classmethod
    def evaluate_training(cls):
        print("evaluating training")
        cols = MODEL_COLS

        classifier = cls.classifier
        predicted_act = classifier.predict(cls.test[cols]) # array of "labels"

        acc_score = accuracy_score(predicted_act, cls.test[["activity"]])
        print("ACC SCORE:", acc_score)
        print("CLASSIFICATION REPORT:\n", classification_report(cls.test[["activity"]], predicted_act))
        print("CONFUSION MATRIX:\n", confusion_matrix(cls.test[['activity']], predicted_act, labels=[0, 1, 2, 3]))


    @classmethod
    def predict_activity(cls, sample:pd.DataFrame):
        """Predicts activities using a sample of preprocessed sensor-data"""
        classifier = cls.classifier
        prediction = classifier.predict(sample)
        return prediction


def train_model():
    """Train and test the model using existing data sets"""
    Recognizer.read_data()
    Recognizer.train_model()
    Recognizer.evaluate_training()
    return True

def use_model(input_data:list[list]) -> str:
    """Uses the trained model to predict the exercize being performed.
    Returns the activity (if not too ambigous) as a string."""

    result: str

    if input_data != None:
        # Create a df with the recieved data
        sample = input_data
        sample = pd.DataFrame(sample, columns=['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
        
        # PREPROCESS DATA (same as training data)
        sample = Recognizer.preprocess_data(sample)
        sample = Recognizer.feature_extraction(sample)
        # print("sample\n", sample.head(), sample.shape)

        # PREDICTION
        prediction = Recognizer.predict_activity(sample)
        # print("PREDICT: ", prediction)

        # PARSE THE PREDICTION
        prediction_count = collections.Counter(prediction) # majority = first idx
        print(prediction_count, list(prediction_count.keys()))
        act_id, count = prediction_count.most_common(1)[0] # gets id (and count)
        # (see: https://stackoverflow.com/a/28663910)
        
        # return the detected exercise, if the prediction wasn't too ambigous
        not_clear =  len(sample)*0.4
        if count > not_clear: 
            for key, value in ACT_IDS.items():
                if value == act_id:
                    print(f"Predicted Activity == {key} ({act_id})")
                    result = key # return the name
        else: 
            print("undecided", count, "<", not_clear)
            result = "..."
    else:
        print("No data recieved")
        result = "..."
    return result


# for testing
if __name__ == "__main__":
    train_model()