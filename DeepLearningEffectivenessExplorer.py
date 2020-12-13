from torch import nn
import torch.functional as F
from torch.utils.data import Dataset
import random
import numpy as np

class ExplorereDataset(Dataset):
    def __init__(self):
        self.num_datapoint = 1000
        self.data = self.__create_dataset(self.num_datapoint)

    def __create_dataset(self, numDataPoint):
        dataset = []
        for i in range(numDataPoint):
            rgb = random.randint(0, 255)
            nir = random.randint(0, 255)
            ndvi = (nir - rgb) / (rgb + nir)
            dataset.append([rgb, nir, ndvi])
        return np.array(dataset)

    def __len__(self):
        return self.num_datapoint

    def __getitem__(self, index):
        X, y = self.data[index][0:2], self.data[index][2]
        return X, y

class ExplorerModel(nn.Module):
    def __init__(self):
        self.fc1 = nn.Liear(2, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

