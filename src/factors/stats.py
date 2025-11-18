import pandas as pd
import numpy as np

def add_stat_factors(df, rolling: int = 20):
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)

    if df["Volume"].nunique() <= 1:
        df["roll_corr"] = 0.0
    else:
        try:
            df["roll_corr"] = (
                df["Close"]
                .rolling(rolling)
                .corr(df["Volume"].rolling(rolling).mean())
            )
        except Exception:
            df["roll_corr"] = 0.0

    df["vol_mean"] = df["Close"].rolling(rolling).std()
    df["vol_ratio"] = df["Close"] / df["Close"].rolling(rolling).mean()
    df = df.fillna(0)

    return df
