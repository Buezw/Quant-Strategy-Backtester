import pandas as pd

def zscore_strategy(
    df: pd.DataFrame,
    window: int = 20,
    z_entry: float = 2.0
) -> pd.DataFrame:

    df = df.copy()
    price = df["Close"]

    ma = price.rolling(window).mean()
    std = price.rolling(window).std()

    df["zscore"] = (price - ma) / std
    df["zscore"].bfill(inplace=True)

    df["signal_raw"] = 0
    df.loc[df["zscore"] < -z_entry, "signal_raw"] = 1
    df.loc[df["zscore"] >  z_entry, "signal_raw"] = -1

    df["signal"] = df["signal_raw"].replace(0, pd.NA).ffill().fillna(0).astype(float)

    return df
