import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


class DataPrep:
    def __init__(self, data):
        self.cols_to_impute_mean = []
        self.cols_to_impute_mode = ['prop_review_count', 'prop_review_score', 'srch_adults_cnt', 'srch_children_cnt']
        self.cols_to_impute_median = []
        self.cols_to_drop = ['prop_price_without_discount_local', 'srch_hcom_destination_id',
                             'prop_price_without_discount_usd', 'srch_dest_latitude', 'srch_dest_longitude',
                             'srch_mobile_app', 'srch_visitor_id', 'srch_co', 'srch_date_time', 'srch_mobile_bool'],
        self.missing_regions_dict = {'CASORIA': 'CAMPANIA', 'MARIGLIANO': 'CAMPANIA', 'OTTAVIANO': 'CAMPANIA',
                                     'PAMPLONA': 'NAVARRA', 'POMPEI': 'CAMPANIA', 'BURLADA': 'NAVARRE',
                                     'POZZUOLI': 'CAMPANIA'}
        self.cols_to_purge_top_percentile = ['prop_price_with_discount_local', 'prop_price_with_discount_usd']
        self.cols_to_fill = {'srch_visitor_wr_member': 'Unknown', 'srch_posa_continent': 'NORTH AMERICA'}
        self.cols_to_purge_bottom_percentile = []
        self.numerical_cols = ['prop_price_with_discount_local', 'prop_price_with_discount_usd', 'prop_imp_drr',
                               'prop_review_score', 'prop_review_count']
        self.categorical_cols = ['srch_visitor_loc_country', 'srch_visitor_loc_region', 'srch_visitor_loc_city',
                                 'srch_visitor_wr_member', 'srch_posa_continent', 'srch_posa_country', 'srch_device',
                                 'srch_currency', 'prop_super_region', 'prop_continent', 'prop_country',
                                 'srch_visitor_visit_nbr', 'srch_los', 'srch_bw', 'srch_adults_cnt',
                                 'srch_children_cnt', 'srch_rm_cnt', 'srch_ci_day', 'srch_co_day',
                                 'prop_travelad_bool', 'prop_dotd_bool', 'prop_brand_bool',
                                 'prop_starrating', 'prop_market_id', 'prop_submarket_id', 'prop_room_capacity',
                                 'prop_hostel_bool']
        self.datetime_cols = ['srch_ci', 'srch_local_date'] #'srch_date_time', 'srch_co'
        self.data = data
        self.data.set_index('srch_id', inplace=True)
        self.data = self.parse_timestamps()
        self.data = self.drop_cols_and_empty_rows_and_fill()
        self.data = self.purge_outliers()
        self.data = self.impute_room_capacity_median()
        self.data = self.impute_averages()
        self.posa_posu_vector_dict = self.generate_posa_posu_vectors(self.data)
        self.data = self.fill_regions(self.data)
        self.data = self.impute_currency_from_vectors(self.data)
        self.data.reset_index().set_index(['srch_id', 'prop_key'], inplace=True)
        self.data = self.encode()
        self.data = self.standardise_and_normalise()

    def return_df(self):
        return self.data

    def parse_timestamps(self):
        self.data['srch_date_time'] = pd.to_datetime(self.data['srch_date_time'], format='%Y-%m-%d %H:%M:%S')
        self.data['srch_ci'] = pd.to_datetime(self.data['srch_ci'], format='%Y-%m-%d')
        self.data['srch_co'] = pd.to_datetime(self.data['srch_co'], format='%Y-%m-%d')
        self.data['srch_local_date'] = pd.to_datetime(self.data['srch_local_date'], format='%Y-%m-%d')
        return self.data

    def drop_cols_and_empty_rows_and_fill(self):
        for col in self.cols_to_drop:
            self.data = self.data.drop(col, axis=1)
        self.data = self.data.dropna(how='all')
        for col in self.cols_to_fill.keys():
            self.data[col] = self.data[col].fillna(self.cols_to_fill[col])
        return self.data

    def purge_outliers(self):
        for col in self.cols_to_purge_top_percentile:
            self.data = self.data[(self.data[col] < self.data[col].quantile(.99))]
        for col in self.cols_to_purge_bottom_percentile:
            self.data = self.data[(self.data[col] > self.data[col].quantile(.01))]
        return self.data

    def fill_regions(self, df):
        for index, row in df.iterrows():
            if str(row['srch_visitor_loc_region']) == 'nan':
                df.loc[index, 'srch_visitor_loc_region'] = self.missing_regions_dict[row['srch_visitor_loc_city']]
        return df

    def generate_posa_posu_vectors(self, df):
        currency_popularity_dict = {}
        currency_popularity = df[df['srch_currency'].notnull()][
            ['srch_posa_country', 'prop_country', 'srch_currency']].reset_index().drop('srch_id', axis=1).groupby(
            ['srch_posa_country', 'prop_country', 'srch_currency']).size().reset_index()
        for posa in currency_popularity['srch_posa_country'].unique():
            posa_df = currency_popularity[currency_popularity['srch_posa_country'] == posa]
            for posu in posa_df['prop_country'].unique():
                currency_popularity_dict[(posa, posu)] = posa_df[posa_df['prop_country'] == posu]\
                    .reset_index()['srch_currency'][0]
        return currency_popularity_dict

    def impute_currency_from_vectors(self, df):
        # todo replace dummy imputer
        df['srch_currency'] = df['srch_currency'].fillna('USD')
        # for index, row in df[df['srch_currency'].isna()].iterrows():
        #     try:
        #         df.loc[index, 'srch_currency'] = self.posa_posu_vector_dict[row['srch_posa_country'] + '_' +
        #                                                                     row['prop_country']]
        #     except KeyError:
        #         df.loc[index, 'srch_currency'] = 'USD'
        return df

    def split_rewards_col(self):
        # todo
        return self.data

    def split_timestamp_columns(self):
        # todo
        return self.data

    def impute_averages(self):
        mean_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        median_imp = SimpleImputer(missing_values=np.nan, strategy='median')
        # mode_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        for col in self.cols_to_impute_mean:
            self.data[col] = mean_imp.fit_transform(self.data[col])
        for col in self.cols_to_impute_median:
            self.data[col] = median_imp.fit_transform(self.data[col])
        for col in self.cols_to_impute_mode:
            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
        return self.data

    def impute_room_capacity_median(self):
        self.data.loc[self.data['prop_room_capacity'] < 0, 'prop_room_capacity'] = \
            self.data['prop_room_capacity'].median()
        return self.data

    def encode(self):
        label_encoder = preprocessing.LabelEncoder()
        ohe = preprocessing.OneHotEncoder(sparse=False)
        for col in self.categorical_cols:
            self.data[col] = label_encoder.fit_transform(self.data[col].map(str))
        for col in self.datetime_cols:
            self.data[col] = label_encoder.fit_transform(self.data[col].map(str))
        return self.data

    def standardise_and_normalise(self):
        self.scaler = StandardScaler()
        self.normaliser = Normalizer(norm='l2')
        for col in self.numerical_cols:
            if col not in ('prop_booking_bool', 'prop_key'):
                self.data[[col]] = self.scaler.fit_transform(self.data[[col]])
                self.data[col] = self.normaliser.fit_transform(self.data[col].reset_index())
        return self.data

    def inverse_scaler(self, df, col):
        df[col] = self.scaler.inverse_transform(df[col])
        return df