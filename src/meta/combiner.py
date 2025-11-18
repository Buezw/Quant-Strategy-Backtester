import numpy as np
import pandas as pd

def combine_signals(df, strat_cols, weight_matrix):
    W = weight_matrix
    S = df[strat_cols].values

    meta_signal = (W * S).sum(axis=1)

    df["meta_signal"] = meta_signal
    return df
