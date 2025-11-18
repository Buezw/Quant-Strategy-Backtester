import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class MetaSequenceDataset(Dataset):


    def __init__(self, df, strat_cols, factor_cols, horizon=1, seq_len=32):

        self.seq_len = seq_len
        self.horizon = horizon

        self.strat_cols = strat_cols
        self.factor_cols = factor_cols

        self.df = df.reset_index(drop=True)

        self.y = df["Close"].pct_change().shift(-horizon).fillna(0).values

    def __len__(self):
        return len(self.df) - self.seq_len - self.horizon

    def __getitem__(self, idx):

        seq = self.df[self.factor_cols].iloc[idx : idx + self.seq_len].values
        seq = seq.astype(np.float32)


        signals_now = self.df[self.strat_cols].iloc[idx + self.seq_len].values.astype(np.float32)

        # label (float32)
        y = np.float32(self.y[idx + self.seq_len])

        return seq, signals_now, y
