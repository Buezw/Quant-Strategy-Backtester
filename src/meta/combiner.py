# src/ensemble/combiner.py
import numpy as np
import pandas as pd

def combine_signals(df, strat_cols, weight_matrix):
    """
    df: 包含策略信号的 DataFrame
    strat_cols: ["sig_ma","sig_rsi","sig_macd","sig_boll"]
    weight_matrix: (N, num_strats) 来自 Transformer 输出的权重
    """
    W = weight_matrix
    S = df[strat_cols].values

    meta_signal = (W * S).sum(axis=1)

    df["meta_signal"] = meta_signal
    return df
