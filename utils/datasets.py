#  Copyright (c) 2022 Andrew
#  Email: andrewlee1807@gmail.com
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# List dataset name
cnu_str = "CNU"
cnu_str_engineering_7 = "CNU_ENGINEERING_7"
cnu_str_engineering_3 = "CNU_ENGINEERING_3"
cnu_str_agriculture_3 = "CNU_AGRICULTURE_3"
cnu_str_agriculture_4 = "CNU_AGRICULTURE_4"
cnu_str_headquarter = "CNU_HEADQUARTER"
cnu_str_truth = "CNU_TRUTH"
comed_str = "COMED"
spain_str = "SPAIN"
france_household_hour_str = "FRANCE_HOUSEHOLD_HOUR"
gyeonggi_str = "GYEONGGI"
gyeonggi9654_str = "GYEONGGI9654"
gyeonggi6499_str = "GYEONGGI6499"
gyeonggi2955_str = "GYEONGGI2955"

# Dataset path
CONFIG_PATH = {
    # cnu_str: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/%EA%B3%B5%EB%8C%807%ED%98%B8%EA%B4%80_HV_02.csv",
    cnu_str: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/%EA%B3%B5%EB%8C%807%ED%98%B8%EA%B4%80_HV_02_datetime.csv",
    cnu_str_engineering_7: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/cnu_multivariable/Engineering-building-7.csv",
    cnu_str_engineering_3: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/cnu_multivariable/Engineering-building-3.csv",
    cnu_str_agriculture_3: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/cnu_multivariable/Agricultural-building-3.csv",
    cnu_str_agriculture_4: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/cnu_multivariable/Agricultural-building-4.csv",
    cnu_str_headquarter: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/cnu_multivariable/Headquarter.csv",
    cnu_str_truth: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/cnu_multivariable/Truth-building.csv",
    comed_str: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/COMED_hourly.csv",
    spain_str: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/spain/spain_ec_499.csv",
    france_household_hour_str: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/france_household/france_household_hour_power_consumption.csv",
    gyeonggi_str: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/gyeonggi_univariable/2955_1hour.csv",
    gyeonggi9654_str: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/gyeonggi_univariable/9654_1hour.csv",
    gyeonggi6499_str: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/gyeonggi_univariable/6499_1hour.csv",
    gyeonggi2955_str: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/gyeonggi_multivariable/2955_1hour.csv"
}


class DataLoader(ABC):
    """
    Class to be inheritance from others dataset
    """

    def __init__(self, path_file, data_name, noise_std=0.0, smoothing_window=0):
        self.raw_data = None
        self.noise_std = noise_std
        self.smoothing_window = smoothing_window
        if path_file is None:
            self.path_file = CONFIG_PATH[data_name]
        else:
            self.path_file = path_file
        print("Reading data from: {}".format(self.path_file))

    def read_data_frame(self):
        return pd.read_csv(self.path_file)

    def read_a_single_sequence(self):
        return np.loadtxt(self.path_file)

    @abstractmethod
    def export_a_single_sequence(self):
        pass

    def add_noise(self, data):
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, data.shape)
            return data + noise
        return data

    def smooth_data(self, data):
        if self.smoothing_window > 0:
            return pd.Series(data.flatten()).rolling(window=self.smoothing_window, min_periods=1).mean().to_numpy().reshape(data.shape)
        return data

    def export_the_sequence(self, feature_names):
        data = self.raw_data[feature_names].to_numpy()
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Apply transformations
        data = self.smooth_data(data)
        data = self.add_noise(data)
        
        return data


# GYEONGGI dataset
class GYEONGGI(DataLoader):
    def __init__(self, data_name=None, path_file=None, noise_std=0.0, smoothing_window=0):
        if data_name is None:
            super(GYEONGGI, self).__init__(path_file, gyeonggi_str, noise_std, smoothing_window)
        else:
            super(GYEONGGI, self).__init__(path_file, data_name, noise_std, smoothing_window)
        self.raw_data = self.read_data_frame()

    def read_data_frame(self):
        return pd.read_csv(self.path_file, header=0, sep='\t')

    def export_a_single_sequence(self):
        return self.raw_data['Amount of Consumption'].to_numpy()  # a single sequence


# CNU dataset
class CNU(DataLoader):
    def __init__(self, data_name=None, path_file=None):
        if data_name is None:
            super(CNU, self).__init__(path_file, cnu_str)
        else:
            super(CNU, self).__init__(path_file, data_name)
        self.raw_data = self.read_data_frame()

    # def export_a_single_sequence(self):
    #     return self.raw_data  # a single sequence

    def read_data_frame(self):
        return pd.read_csv(self.path_file, header=0, sep=',')

    def export_a_single_sequence(self):
        try:
            return self.raw_data['전력사용량'].to_numpy()  # a single sequence
        except KeyError:
            try:
                return self.raw_data['energy'].to_numpy()
            except KeyError:
                return self.raw_data['PowerConsumption'].to_numpy()

    def export_the_sequence(self, feature_names):
        if isinstance(feature_names, list):
            return self.raw_data[feature_names].to_numpy()
        return self.raw_data[feature_names].to_numpy().reshape(-1, 1)


# COMED_hourly
class COMED(DataLoader):
    def __init__(self, path_file=None):
        super(COMED, self).__init__(path_file, comed_str)
        self.dataframe = self.read_data_frame()


# Spain dataset
class SPAIN(DataLoader):
    # TODO:
    # Combine all the datasets: POWER Consumption & Weather Data, into one dataset
    def __init__(self, path_file=None, noise_std=0.0, smoothing_window=0):
        super(SPAIN, self).__init__(path_file, spain_str, noise_std, smoothing_window)
        self.raw_data = self.read_data_frame()

    def export_a_single_sequence(self):
        # Pick the customer no 20
        return self.raw_data.loc[:, '20']  # a single sequence

    def export_the_sequence(self, feature_names):
        # must change this function after changing dataset
        if isinstance(feature_names, list):
            return self.raw_data[feature_names].to_numpy()
        return self.export_a_single_sequence().to_numpy().reshape(-1, 1)
        # return self.raw_data[feature_names].to_numpy()


def fill_missing(data):
    one_day = 23 * 60
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            if np.isnan(data[row, col]):
                data[row, col] = data[row - one_day, col]


class FRANCEHOUSEHOLD(DataLoader):
    def __init__(self, path_file=None, noise_std=0.0, smoothing_window=0):
        super(FRANCEHOUSEHOLD, self).__init__(path_file, france_household_hour_str, noise_std, smoothing_window)
        self.raw_data = self.read_data_frame()

    def read_data_frame(self):
        return pd.read_csv(self.path_file, header=0, sep=',')

    def export_a_single_sequence(self):
        if not hasattr(self, 'data_by_hour'):
            self.load_data()
        return self.data_by_hour['Global_active_power'].to_numpy()

    def load_data(self):

        df = pd.read_csv(self.path_file,
                         parse_dates=['dt'],
                         infer_datetime_format=True,
                         low_memory=False, na_values=['nan', '?'],
                         index_col='dt')

        droping_list_all = []
        for j in range(0, 7):
            if not df.iloc[:, j].notnull().all():
                droping_list_all.append(j)
        for j in range(0, 7):
            df.iloc[:, j] = df.iloc[:, j].fillna(df.iloc[:, j].mean())

        fill_missing(df.values)

        self.df = df
        self.data_by_days = df.resample('D').sum()  # all the units of particular day
        self.data_by_hour = df.resample('H').sum()  # all the units of particular day
