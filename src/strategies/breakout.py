import pandas as pd

def breakout_strategy(
    df: pd.DataFrame,
    high_window: int = 20,
    low_window: int = 10
) -> pd.DataFrame:
    df = df.copy()

    df["highest"] = df["High"].rolling(high_window).max()
    df["lowest"]  = df["Low"].rolling(low_window).min()


    df[["highest", "lowest"]] = df[["highest", "lowest"]].bfill()
    df["signal_raw"] = 0
    df.loc[df["Close"] > df["highest"], "signal_raw"] = 1
    df.loc[df["Close"] < df["lowest"],  "signal_raw"] = -1

    df["signal"] = df["signal_raw"].replace(0, pd.NA).ffill().fillna(0).astype(float)

    return df
