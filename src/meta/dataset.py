# src/meta/dataset.py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class MetaSequenceDataset(Dataset):
    """
    构造 Transformer 训练用的序列数据：
    - X_seq: (seq_len, num_features)
    - signals_now: (num_strategies,)
    - y: 未来 horizon 的真实 return
    """

    def __init__(self, df, strat_cols, factor_cols, horizon=1, seq_len=32):
        """
        df: 已包含因子+所有策略 signal 的 DataFrame
        strat_cols: 策略 signal 列表，例如 ["sig_ma","sig_rsi","sig_macd","sig_boll"]
        factor_cols: 因子列
        """
        self.seq_len = seq_len
        self.horizon = horizon

        self.strat_cols = strat_cols
        self.factor_cols = factor_cols

        self.df = df.reset_index(drop=True)

        # 未来 horizon 的真实收益
        self.y = df["Close"].pct_change().shift(-horizon).fillna(0).values

    def __len__(self):
        return len(self.df) - self.seq_len - self.horizon

    def __getitem__(self, idx):
        # 过去 seq_len 的序列数据
        seq = self.df[self.factor_cols].iloc[idx : idx + self.seq_len].values
        seq = seq.astype(np.float32)

        # 当前策略信号
        signals_now = self.df[self.strat_cols].iloc[idx + self.seq_len].values.astype(np.float32)

        # label (float32)
        y = np.float32(self.y[idx + self.seq_len])

        return seq, signals_now, y
