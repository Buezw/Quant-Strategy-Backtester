import pandas as pd
import numpy as np


def ma_crossover(
    df,
    short=10,
    long=50,
    min_gap=0.001,       # MA 之间至少差 0.1% 才算有效信号
    confirm_bars=1,      # 信号确认 K 数
    volume_filter=False, # 是否启用成交量过滤
):
    """Advanced MA Crossover
    - min_gap: 过滤假突破
    - confirm_bars: 信号确认
    - volume_filter: 成交量过滤（可选）
    """

    df = df.copy()

    # MA
    df["ma_short"] = df["Close"].rolling(short).mean()
    df["ma_long"] = df["Close"].rolling(long).mean()

    # MA 差值（过滤假突破）
    df["ma_diff"] = df["ma_short"] - df["ma_long"]
    df["ma_gap"]  = df["ma_diff"] / df["ma_long"]

    # 初步信号
    df["raw_signal"] = 0
    df.loc[df["ma_gap"] > min_gap, "raw_signal"] = 1
    df.loc[df["ma_gap"] < -min_gap, "raw_signal"] = -1

    # 信号确认（连续 confirm_bars 个 bar 方向一致）
    df["signal"] = (
        df["raw_signal"]
        .rolling(confirm_bars)
        .apply(lambda x: x.iloc[-1] if (np.all(x == x.iloc[-1]) and x.iloc[-1] != 0) else 0)
        .fillna(0)
    )

    # 成交量过滤（可选，用于 crypto 或高频）
    if volume_filter and "Volume" in df.columns:
        vol_ma = df["Volume"].rolling(long).mean()
        df.loc[df["Volume"] < vol_ma, "signal"] = 0

    # 前几行无 MA → 无信号
    df["signal"] = df["signal"].fillna(0)

    return df
