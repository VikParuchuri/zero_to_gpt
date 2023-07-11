import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import os
import gdown
import pathlib

DATA_DIR = pathlib.Path(__file__).parent.resolve()


class CSVDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx]).float()
        y = torch.from_numpy(self.y[idx]).float()
        return x, y


class CSVDatasetWrapper:
    predictors = None
    target = None
    file_name = None
    splits = ["train", "validation", "test"]
    download_link = None
    dataset_cls = CSVDataset

    def __init__(self):

        fpath = os.path.join(DATA_DIR, self.file_name)

        if not os.path.exists(fpath):
            gdown.download(self.download_link, fpath)

        data = pd.read_csv(fpath)
        self.data = data
        self.clean_data()
        # Ensure we get same split every time
        np.random.seed(0)
        self.split_x_target()
        self.create_final_sets()

    def clean_data(self):
        """Optionally implement in subclass"""
        pass

    def split_x_target(self):
        """
        Optionally implement in subclass
        """
        all_splits = {}
        split_data = np.split(self.data, [int(.7 * len(self.data)), int(.85 * len(self.data))])
        splits = [
            [d[self.predictors].to_numpy(), d[[self.target]].to_numpy()] for d in
            split_data]

        for split_name, split in zip(self.splits, splits):
            all_splits[split_name] = {"x": split[0], "target": split[1]}

        self.split_data = all_splits

    def create_final_sets(self):
        """
        Optional override to do final processing
        """
        self.final_data = self.split_data

    def generate_dataset(self, x, target, batch_size):
        dataset = self.dataset_cls(x, target)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader

    def generate_datasets(self, batch_size):
        np.random.seed(0)

        datasets = {}
        for split_name, split in self.final_data.items():
            loader = self.generate_dataset(split["x"], split["target"], batch_size)
            datasets[split_name] = loader

        return datasets

    def get_flat_datasets(self):
        datasets = []
        for split_name, split in self.final_data.items():
            datasets.append((split["x"], split["target"]))
        return datasets


class HousePricesDatasetWrapper(CSVDatasetWrapper):
    predictors = ["interest", "vacancy", "cpi", "price", "value", "adj_price", "adj_value"]
    target = "next_quarter"
    file_name = "house_prices.csv"
    splits = ["train", "validation", "test"]
    download_link = "https://drive.google.com/uc?export=download&id=1eWeeYlxNatpXQSsiFwQg2HJeYCTJNj9Z"

    def clean_data(self):
        self.scaler = StandardScaler()
        data = self.data.ffill()
        data[self.predictors] = self.scaler.fit_transform(self.data[self.predictors])
        data[self.target] = (data[self.target] - data[self.target].min()) // 1000
        self.data = data


class WeatherDatasetWrapper(CSVDatasetWrapper):
    predictors = ["tmax", "tmin", "rain"]
    target = "tmax_tomorrow"
    file_name = "clean_weather.csv"
    splits = ["train", "validation", "test"]
    download_link = "https://drive.google.com/uc?export=download&id=1O_uOTvMJb2FkUK7rB6lMqpPQqiAdLXNL"
    sequence_length = 7

    def clean_data(self):
        self.scaler = StandardScaler()
        data = self.data.ffill()
        data[self.predictors] = self.scaler.fit_transform(data[self.predictors])
        self.data = data


class WeatherDatasetWrapperRNN(CSVDatasetWrapper):
    predictors = ["tmax", "tmin", "rain"]
    target = "tmax_tomorrow"
    file_name = "clean_weather.csv"
    splits = ["train", "validation", "test"]
    download_link = "https://drive.google.com/uc?export=download&id=1O_uOTvMJb2FkUK7rB6lMqpPQqiAdLXNL"
    sequence_length = 7

    def clean_data(self):
        self.scaler = StandardScaler()
        data = self.data.ffill()
        data[self.predictors] = self.scaler.fit_transform(data[self.predictors])
        self.data = data

    def create_final_sets(self):
        final_data = {}
        for split_name, split in self.split_data.items():
            for j in range(0, len(split["x"]) - self.sequence_length):
                if split_name not in final_data:
                    final_data[split_name] = {"x": [], "target": []}
                final_data[split_name]["x"].append(split["x"][j:(j+7)])
                final_data[split_name]["target"].append(split["target"][j:(j+7)])
        self.final_data = final_data


class SkyServerDatasetWrapper(CSVDatasetWrapper):
    predictors = ["ra", "dec", "u", "g", "r", "i", "z", "run", "camcol", "field", "redshift", "plate", "mjd"]
    target = "class"
    file_name = "skyserver.csv"
    splits = ["train", "validation", "test"]
    download_link = "https://drive.google.com/u/0/uc?id=1gYwg5YyaV3zUX-07bLCol8E0M-uX__zX&export=download"

    def clean_data(self):
        self.scaler = StandardScaler()
        data = self.data.ffill()
        data[self.predictors] = self.scaler.fit_transform(data[self.predictors])
        data[self.target] = data[self.target].replace({"STAR": 0, "GALAXY": 1, "QSO": 2})
        self.data = data


class SkyServerBinaryDatasetWrapper(SkyServerDatasetWrapper):
    def clean_data(self):
        self.scaler = StandardScaler()
        data = self.data.ffill()
        data[self.predictors] = self.scaler.fit_transform(data[self.predictors])
        data[self.target] = data[self.target].replace({"STAR": 0, "GALAXY": 1, "QSO": 0})
        self.data = data
