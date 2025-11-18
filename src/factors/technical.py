import pandas as pd
import numpy as np

def add_technical_factors(df: pd.DataFrame):
    df["ret1"] = df["Close"].pct_change()
    df["ma5"]  = df["Close"].rolling(5).mean()
    df["ma10"] = df["Close"].rolling(10).mean()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["momentum10"] = df["Close"].pct_change(10)
    return df
