import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Analyser:
    def __init__(self, data):
        self.cols_to_impute_mean = []
        self.cols_to_impute_mode = []
        self.cols_to_impute_median = []

        self.data = data
        self.data = self.parse_timestamps()
        self.data = self.drop_cols_and_empty_rows()
        self.data = self.impute()
        self.data = self.standardise()

    def parse_timestamps(self):
        self.data = pd.to_datetime(self.data['srch_date_time'], format='%Y/%m/%d %H:%M:%ss')
        self.data = pd.to_datetime(self.data['srch_ci'], format='%Y/%m/%d')
        self.data = pd.to_datetime(self.data['srch_co'], format='%Y/%m/%d')
        self.data = pd.to_datetime(self.data['srch_local_date'], format='%Y/%m/%d')
        return self.data

    def drop_cols_and_empty_rows(self):
        self.data = self.data.dropna(how='all')
        self.data = self.data.drop('srch_visitor_id', axis=1)
        return self.data

    def impute(self):
        median_imp = Imputer(missing_values='NaN', strategy='median', axis=0)
        return self.data

    def standardise(self):
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)
        return self.data

    def label_encoder(self):
        label_encoder = preprocessing.LabelEncoder()
        for col in cols:
            self.data = label_encoder.fit_transform(self.data[col].map(str))
        return self.data