import pandas as pd

def momentum_strategy(
    df: pd.DataFrame,
    lookback: int = 20
) -> pd.DataFrame:

    df = df.copy()

    df["momentum"] = df["Close"] - df["Close"].shift(lookback)
    df["momentum"].bfill(inplace=True)

    df["signal_raw"] = 0
    df.loc[df["momentum"] > 0, "signal_raw"] = 1
    df.loc[df["momentum"] < 0, "signal_raw"] = -1

    df["signal"] = df["signal_raw"].replace(0, pd.NA).ffill().fillna(0).astype(float)

    return df
