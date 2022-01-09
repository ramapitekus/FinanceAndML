import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, x, y, sequence_length=3):
        self.sequence_length = sequence_length
        self.y = y
        self.x = x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.x[i_start : (i + 1), :]
        else:
            padding = self.x[0].repeat(self.sequence_length - i - 1, 1)
            x = self.x[0 : (i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]


class StandardDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        return x, y
