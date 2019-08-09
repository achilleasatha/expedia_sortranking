import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataPrep:
    def __init__(self, data):
        self.cols_to_impute_mean = []
        self.cols_to_impute_mode = ['prop_review_count', 'prop_review_score', 'srch_adults_cnt', 'srch_children_cnt']
        self.cols_to_impute_median = []
        self.cols_to_drop = ['prop_price_without_discount_local', 'srch_hcom_destination_id',
                             'prop_price_without_discount_usd', 'srch_dest_latitude', 'srch_dest_longitude',
                             'srch_mobile_app'],
        self.cols_to_impute_from_other_cols = {'srch_currency': 'srch_posa_country',
                                               'srch_visitor_loc_city':'srch_posa_country'}
        self.cols_to_impute_from_vectors = ['prop_price_with_discount_local', 'prop_price_with_discount_usd']
        self.cols_to_purge_top_percentile = ['prop_price_with_discount_local', 'prop_price_with_discount_usd']
        self.cols_to_purge_bottom_percentile = []
        self.data = data
        self.data.set_index('srch_id', inplace=True)
        self.data = self.parse_timestamps()
        self.data = self.drop_cols_and_empty_rows()
        self.data = self.purge_outliers()
        self.data = self.fix_room_capacity()
        self.generate_posa_region_map()
        self.data = self.fill_regions()
        self.generate_posa_posu_vectors()
        self.data = self.impute_mean_from_vector()
        self.data = self.impute_averages()
        self.data = self.standardise()

    def return_df(self):
        return self.data

    def parse_timestamps(self):
        self.data['srch_date_time'] = pd.to_datetime(self.data['srch_date_time'], format='%Y-%m-%d %H:%M:%S')
        self.data['srch_ci'] = pd.to_datetime(self.data['srch_ci'], format='%Y-%m-%d')
        self.data['srch_co'] = pd.to_datetime(self.data['srch_co'], format='%Y-%m-%d')
        self.data['srch_local_date'] = pd.to_datetime(self.data['srch_local_date'], format='%Y-%m-%d')
        return self.data

    def drop_cols_and_empty_rows(self):
        for col in self.cols_to_drop:
            self.data = self.data.drop(col, axis=1)
        self.data = self.data.dropna(how='all')
        return self.data

    def purge_outliers(self):
        for col in self.cols_to_purge_top_percentile:
            self.data = self.data[(self.data[col] < self.data[col].quantile(.99))]
        for col in self.cols_to_purge_bottom_percentile:
            self.data = self.data[(self.data[col] > self.data[col].quantile(.01))]
        return self.data

    def generate_posa_region_map(self):
        pass
        # todo
        # return posa_region_map

    def fill_regions(self):
        # todo
        return self.data

    def generate_posa_posu_vectors(self):
        pass
        # todo ['srch_posa_country', 'prop_country']
        # return posa_posu_vectors

    def impute_mean_from_vector(self):
        # todo ['srch_posa_country', 'prop_country']
        for cols in self.cols_to_impute_from_vectors:
            pass
        return self.data

    def impute_averages(self):
        mean_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        median_imp = SimpleImputer(missing_values=np.nan, strategy='median')
        mode_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        for col in self.cols_to_impute_mean:
            self.data[col] = mean_imp.fit_transform(self.data[col])
        for col in self.cols_to_impute_median:
            self.data[col] = median_imp.fit_transform(self.data[col])
        for col in self.cols_to_impute_mode:
            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
        return self.data

    def fix_room_capacity(self):
        self.data[self.data['prop_room_capacity'] < 0] = self.data['prop_room_capacity'].median()
        return self.data

    def standardise(self):
        scaler = StandardScaler()
        # self.data = scaler.fit_transform(self.data)
        return self.data

    def label_encoder(self):
        label_encoder = preprocessing.LabelEncoder()
        # for col in self.data.columns:
        #     self.data = label_encoder.fit_transform(self.data[col].map(str))
        return self.data