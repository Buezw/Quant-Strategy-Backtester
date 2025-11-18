import pandas as pd

def bollinger_strategy(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    price_col: str = "Close"
) -> pd.DataFrame:
    df = df.copy()
    price = df[price_col]

    ma = price.rolling(window).mean()
    std = price.rolling(window).std()

    df["bb_mid"]   = ma
    df["bb_upper"] = ma + num_std * std
    df["bb_lower"] = ma - num_std * std


    df["signal_raw"] = 0
    df.loc[price < df["bb_lower"], "signal_raw"] = 1
    df.loc[price > df["bb_upper"], "signal_raw"] = -1


    df["signal"] = df["signal_raw"].replace(0, pd.NA).ffill().fillna(0).astype(float)

    return df
