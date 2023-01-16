import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

class WeatherDatasetWrapper:
    predictors = ["tmax", "tmin", "rain"]
    target = "tmax_tomorrow"
    file_name = "clean_weather.csv"
    splits = ["train", "validation", "test"]
    sequence_length = 7

    def __init__(self, device):
        self.device = device
        self.scaler = StandardScaler()

        data = pd.read_csv("../../data/clean_weather.csv")
        data = data.ffill()
        data[self.predictors] = self.scaler.fit_transform(data[self.predictors])

        self.data = data

    def generate_dataset(self, x, target, batch_size):
        dataset = RegressionDataset(x, target, self.device, self.sequence_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader

    def generate_datasets(self, batch_size):
        np.random.seed(0)

        split_data = np.split(self.data, [int(.7 * len(self.data)), int(.85 * len(self.data))])
        splits = [
            [torch.from_numpy(d[self.predictors].to_numpy()), torch.from_numpy(d[[self.target]].to_numpy())] for d in split_data]

        datasets = {}
        for split_name, split in zip(self.splits, splits):
            loader = self.generate_dataset(split[0], split[1], batch_size)
            datasets[split_name] = loader

        return datasets


class RegressionDataset(Dataset):
    def __init__(self, x, y, device, sequence_length):
        self.dataset = [x,y]
        self.sequence_length = sequence_length
        self.device = device

    def __len__(self):
        return len(self.dataset[0]) - self.sequence_length

    def __getitem__(self, idx):
        x, y = self.dataset[0][idx:(idx+self.sequence_length)], self.dataset[1][idx:(idx+self.sequence_length)]
        return x.float().to(self.device), y.float().to(self.device)