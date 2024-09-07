import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

pd.set_option('future.no_silent_downcasting', True)

datasets = {"WebTraffic": "WebTrafficLAcity/lacity.org-website-traffic.csv",
            "StoreItems": "StoreItemandDemandForecastingChallenge/train.csv",
            "AustraliaTourism": "QuarterlyTourismAustralia/tourism.csv",
            "MetroTraffic": "MetroInterstateTrafficVolume/Metro_Interstate_Traffic_Volume.csv/Metro_Interstate_Traffic_Volume.csv",
            "BeijingAirQuality": "BeijingAirQuality/beijing+multi+site+air+quality+data"}


class CyclicEncoder:

    def __init__(self, name, df, propCycEnc):
        self.column_name = name
        self.categories = df[name].unique()
        counts = df[name].value_counts(dropna=False)
        total_counts = counts.sum()
        angles = (counts/total_counts) * 2 * np.pi
        cumulative_angles = angles.cumsum() - (angles / 2)
        temp = counts.index.values
        """
        counts = df[name].value_counts(dropna=False)

        # Step 2: Calculate the proportional angles (in radians)
        total_counts = counts.sum()
        angles = (counts / total_counts) * 2 * np.pi  # Proportional angles in radians

        # Step 3: Calculate the cumulative angle positions
        cumulative_angles = angles.cumsum() - (angles / 2)
        """
        self.categories = temp
        if propCycEnc:
            self.angles = cumulative_angles
        else:
            self.angles = np.array(list(range(len(self.categories)))) * (2 * np.pi) / len(self.categories)
        self.mapper = dict(zip(self.categories, self.angles))
        self.mapper_sine = dict(zip(self.categories, np.sin(self.angles)))
        self.mapper_cosine = dict(zip(self.categories, np.cos(self.angles)))
        self.angles_to_cat = dict(zip(self.angles, self.categories))

    def encode(self, df):
        df_copy = df.copy()
        df_copy[self.column_name + "_sine"] = df_copy[self.column_name].replace(self.mapper_sine).astype(float)
        df_copy[self.column_name + "_cos"] = df_copy[self.column_name].replace(self.mapper_cosine).astype(float)
        df_copy.drop(columns=[self.column_name], inplace=True)
        return df_copy

    def decode(self, df):
        df_copy = df.copy()
        df_copy[self.column_name + "_sine"] = np.clip(df_copy[self.column_name + "_sine"], -1, 1)
        df_copy[self.column_name + "_cos"] = np.clip(df_copy[self.column_name + "_cos"], -1, 1)
        df_copy[self.column_name + "_angle"] = np.nan
        condition1 = np.logical_and(df_copy[self.column_name + "_sine"] >= 0, df_copy[self.column_name + "_cos"] > 0)
        condition2 = np.logical_and(df_copy[self.column_name + "_sine"] > 0, df_copy[self.column_name + "_cos"] <= 0)
        condition3 = np.logical_and(df_copy[self.column_name + "_sine"] <= 0, df_copy[self.column_name + "_cos"] < 0)
        condition4 = np.logical_and(df_copy[self.column_name + "_sine"] < 0, df_copy[self.column_name + "_cos"] >= 0)

        df_copy.loc[condition1, self.column_name + "_angle"] = (np.arcsin(df_copy[self.column_name + "_sine"].values)[
                                                                    condition1.values] +
                                                                np.arccos(df_copy[self.column_name + "_cos"].values)[
                                                                    condition1.values]) / 2

        df_copy.loc[condition2, self.column_name + "_angle"] = (np.arccos(df_copy[self.column_name + "_cos"].values)[
                                                                    condition2.values] +
                                                                np.pi -
                                                                np.arcsin(df_copy[self.column_name + "_sine"].values)[
                                                                    condition2.values]) / 2
        df_copy.loc[condition3, self.column_name + "_angle"] = (2 * np.pi -
                                                                np.arccos(df_copy[self.column_name + "_cos"].values)[
                                                                    condition3.values] +
                                                                np.pi - np.arcsin(
                    df_copy[self.column_name + "_sine"].values)[condition3.values]) / 2
        df_copy.loc[condition4, self.column_name + "_angle"] = (4 * np.pi -
                                                                np.arccos(df_copy[self.column_name + "_cos"].values)[
                                                                    condition4.values] + np.arcsin(
                    df_copy[self.column_name + "_sine"].values)[condition4.values]) / 2

        df_copy[self.column_name + "_angle"] = df_copy[self.column_name + "_angle"] % (2 * np.pi)
        df_copy[self.column_name + '_threshold_angle'] = df_copy[self.column_name + "_angle"].apply(
            lambda x: self.nearest_threshold(x, self.angles))
        df_copy[self.column_name] = df_copy[self.column_name + '_threshold_angle'].replace(self.angles_to_cat)
        df_copy.drop(columns=[self.column_name + '_sine', self.column_name + '_cos', self.column_name + '_angle',
                              self.column_name + '_threshold_angle'], inplace=True)
        return df_copy

    @staticmethod
    def nearest_threshold(x, thresholds):
        return min(thresholds, key=lambda t: abs(t - x))


class Preprocessor:
    def __init__(self, name, propCycEnc):
        self.pce = propCycEnc
        self.cols_to_scale = None
        self.cyclic_encoded_columns = None
        self.encoders = {}
        self.hierarchical_features_uncyclic = []
        self.hierarchical_features_cyclic = []
        self.scaler = StandardScaler()
        self.df_orig = self.fetchDataset(name, False)
        self.column_dtypes = self.df_orig.dtypes.to_dict()
        self.df_cleaned = self.fetchDataset(name, True)

    def fetchDataset(self, name, return_cleaned):
        if name != "BeijingAirQuality":
            df = pd.read_csv(datasets[name])
            if name == "MetroTraffic":
                df['date_time'] = pd.to_datetime(df['date_time'])
                df['year'] = df['date_time'].dt.year
                df['month'] = df['date_time'].dt.month
                df['day'] = df['date_time'].dt.day
                df['hour'] = df['date_time'].dt.hour
                df.drop(columns=['date_time'], inplace=True)
                self.hierarchical_features_uncyclic = ['year', 'month', 'day', 'hour']

        else:
            dfs = []
            csvs = os.listdir(datasets[name])
            for file in csvs:
                dfs.append(pd.read_csv(datasets[name] + "/" + file))
            df = pd.concat(dfs)
            df.drop(columns=['No'], inplace=True)  # redundant

        if return_cleaned:
            df_cleaned = self.cleanDataset(name, df)
            for col in self.hierarchical_features_uncyclic:
                self.hierarchical_features_cyclic.append(col + '_sine')
                self.hierarchical_features_cyclic.append(col + '_cos')
            return df_cleaned

        else:
            return df

    def cleanDataset(self, name, df):
        """Beijing Air Quality has some missing values for the sensor data"""
        df_clean = df.copy()
        if name == "BeijingAirQuality":
            for column in df_clean.columns:
                if df_clean[column].dtype != 'object':
                    df_clean[column] = df_clean[column].interpolate()
            self.cyclic_encoded_columns = ['year', 'month', 'day', 'hour', 'wd', 'station']

        elif name == 'MetroTraffic':
            self.cyclic_encoded_columns = ['year', 'month', 'day', 'hour', 'holiday', 'weather_main',
                                           'weather_description']

        df_cyclic = self.cyclicEncode(df_clean)  # returns the dataframe with cyclic encoding applied

        self.cols_to_scale = [col for col in df_cyclic.columns if
                              col not in self.cyclic_encoded_columns and '_sine' not in col and '_cos' not in col]
        df_cyclic[self.cols_to_scale] = self.scaler.fit_transform(df[self.cols_to_scale])
        return df_cyclic

    def cyclicEncode(self, df):
        df_copy = df.copy()
        for column in self.cyclic_encoded_columns:
            if column not in self.encoders:
                self.encoders[column] = CyclicEncoder(column, df_copy, self.pce)
            df_copy = self.encoders[column].encode(df_copy)
        return df_copy

    def cyclicDecode(self, df):
        df_copy = df.copy()
        for column in self.cyclic_encoded_columns:
            if column + '_sine' not in df_copy.columns:
                continue
            else:
                df_copy = self.encoders[column].decode(df_copy)
                df_copy[column] = df_copy[column].astype(self.column_dtypes[column])

        return df_copy

    def decode(self, dataframe=None, rescale=False):  # without rescaling only the cyclic part is decoded
        df_mod = dataframe.copy()
        for column in self.cyclic_encoded_columns:
            df_mod = self.encoders[column].decode(df_mod)
        if rescale:
            df_mod[self.cols_to_scale] = self.scaler.inverse_transform(df_mod[self.cols_to_scale])

        for col in df_mod.columns:
            df_mod[col] = df_mod[col].astype(self.column_dtypes[col])
        return df_mod

    def scale(self, df):
        df_scaled = df.copy()
        df_scaled[self.cols_to_scale] = self.scaler.transform(df_scaled[self.cols_to_scale])
        return df_scaled

    def rescale(self, df):
        df_rescaled = df.copy()
        df_rescaled[self.cols_to_scale] = self.scaler.inverse_transform(df_rescaled[self.cols_to_scale])
        return df_rescaled


class PreprocessorOneHot:
    def __init__(self, name):
        self.cols_to_scale = None
        self.encoders = {}
        self.scaler = StandardScaler()
        self.hierarchical_features = []
        self.hierarchical_features_onehot = []
        self.onehot_encoded_columns = []
        self.onehot_column_names = []
        self.df_orig = self.fetchDataset(name, False)
        self.column_dtypes = self.df_orig.dtypes.to_dict()
        self.df_cleaned = self.fetchDataset(name, True)

    def fetchDataset(self, name, return_cleaned):
        if name != "BeijingAirQuality":
            df = pd.read_csv(datasets[name])
            if name == "MetroTraffic":
                df['date_time'] = pd.to_datetime(df['date_time'])
                df['year'] = df['date_time'].dt.year
                df['month'] = df['date_time'].dt.month
                df['day'] = df['date_time'].dt.day
                df['hour'] = df['date_time'].dt.hour
                df.drop(columns=['date_time'], inplace=True)
                self.hierarchical_features = ['year', 'month', 'day', 'hour']

        else:
            dfs = []
            csvs = os.listdir(datasets[name])
            for file in csvs:
                dfs.append(pd.read_csv(datasets[name] + "/" + file))
            df = pd.concat(dfs)
            df.drop(columns=['No'], inplace=True)  # redundant

        if return_cleaned:
            df_cleaned = self.cleanDataset(name, df)
            return df_cleaned

        else:
            return df

    def cleanDataset(self, name, df):
        """Beijing Air Quality has some missing values for the sensor data"""
        df_clean = df.copy()
        if name == "BeijingAirQuality":
            for column in df_clean.columns:
                if df_clean[column].dtype != 'object':
                    df_clean[column] = df_clean[column].interpolate()
            self.onehot_encoded_columns = ['year', 'month', 'day', 'hour', 'wd', 'station']

        elif name == 'MetroTraffic':
            self.onehot_encoded_columns = ['year', 'holiday', 'weather_main',
                                           'weather_description']

        df_onehot = self.onehotEncode(df_clean)  # returns the dataframe with cyclic encoding applied

        for feature in self.hierarchical_features:
            if feature in self.onehot_encoded_columns:
                self.hierarchical_features_onehot.extend(self.encoders[feature])
            else:
                self.hierarchical_features_onehot.append(feature)

        self.cols_to_scale = [col for col in df_clean.columns if
                              col not in self.onehot_encoded_columns]
        df_onehot[self.cols_to_scale] = self.scaler.fit_transform(df[self.cols_to_scale])
        return df_onehot

    def onehotEncode(self, df):
        df_copy = df.copy()
        df_copy = pd.get_dummies(df_copy, columns=self.onehot_encoded_columns, dummy_na=True)
        for col in self.onehot_encoded_columns:
            if not df[col].isna().any():
                name = f'{col}_nan'
                df_copy = df_copy.drop(columns=[name])
        if len(self.onehot_column_names) == 0:  # if it's the first time
            self.onehot_column_names = [name for name in df_copy.columns if name not in df.columns]
            for column in self.onehot_encoded_columns:
                names = []
                for ohcs in self.onehot_column_names:
                    if column in ohcs:
                        names.append(ohcs)
                self.encoders[column] = names
        return df_copy

    def onehotDecode(self, df):
        df_copy = df.copy()
        for column in self.encoders.keys():
            df_select = df_copy[self.encoders[column]]
            sep_str = f'{column}_'
            category = pd.from_dummies(df_select, sep=sep_str)
            if self.column_dtypes[column] != 'object':
                category = category.apply(pd.to_numeric)
            df_copy[column] = category.astype(self.column_dtypes[column])
            df_copy = df_copy.drop(columns=self.encoders[column])

        return df_copy

    def decode(self, dataframe=None, rescale=False):  # without rescaling only the cyclic part is decoded
        df_mod = dataframe.copy()
        df_mod = self.onehotDecode(df_mod)
        if rescale:
            df_mod[self.cols_to_scale] = self.scaler.inverse_transform(df_mod[self.cols_to_scale])

        for col in df_mod.columns:
            df_mod[col] = df_mod[col].astype(self.column_dtypes[col])
        return df_mod

    def scale(self, df):
        df_scaled = df.copy()
        df_scaled[self.cols_to_scale] = self.scaler.transform(df_scaled[self.cols_to_scale])
        return df_scaled

    def rescale(self, df):
        df_rescaled = df.copy()
        df_rescaled[self.cols_to_scale] = self.scaler.inverse_transform(df_rescaled[self.cols_to_scale])
        return df_rescaled


