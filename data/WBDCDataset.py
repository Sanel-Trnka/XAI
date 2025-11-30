"""PyTorch dataset wrapper around the tabular WDBC samples."""

from torch.utils.data import Dataset
import torch
from models.common import Config
from pandas import DataFrame

class WBDCDataset(Dataset):
    """Materialize tensors from a pandas frame using the shared preprocessing."""

    def __init__(self, base_df: DataFrame, cfg: Config):
        from utils.common import data_prep
        x_df, y_df = data_prep(base_df, cfg)
        self.xs = torch.from_numpy(x_df.values).float()
        self.ys = torch.from_numpy(y_df.values).long()

    def __getitem__(self, idx):
        """Return the feature tensor and label for a given row index."""
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y

    def __len__(self):
        """Length of the dataset (number of rows)."""
        return len(self.xs)