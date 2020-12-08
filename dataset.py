import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

from open_data import get_df_stats

import torchvision.transforms as transforms


class RadarDataset(Dataset):

    def __init__(self, dataframe,transform=None):
        self.data = dataframe["vol_data_x_norm"].values
        self.target = dataframe["vol_label_y_norm"].values
        self.transform = transform

    def __len__(self):  # total number of samples
        return len(self.data)

    def __getitem__(self, index):  # how to preprocess the files (index = iterator on samples)
        sample = self.data[index]
        label = self.target[index]
        print(label.shape)
        # Normalize your data here
        if self.transform:
            sample = self.transform(sample)
            label=self.transform(label)

        return sample,label


class RadarCollate(object):
    """Function object used as a collate function for DataLoader."""

    def __init__(self, ):
        pass

    def _collate_fn(self, batch):
        data_list, label_list = [], []
        for _data, _label in batch:
            data_list.append(_data)
            label_list.append(_label)
        return torch.Tensor(data_list), torch.LongTensor(label_list)

    def __call__(self, batch):
        return self._collate_fn(batch)


