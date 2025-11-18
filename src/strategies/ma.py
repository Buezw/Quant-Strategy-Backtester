import pandas as pd

def ma_strategy(
    df: pd.DataFrame,
    short_window: int = 10,
    long_window: int = 50,
    price_col: str = "Close"
) -> pd.DataFrame:

    if price_col not in df.columns:
        raise ValueError(f"MA strategy: df 必须包含列 '{price_col}'")

    df = df.copy()
    price = df[price_col]

    df["ma_short"] = price.rolling(short_window).mean()
    df["ma_long"] = price.rolling(long_window).mean()

    df[["ma_short", "ma_long"]] = df[["ma_short", "ma_long"]].bfill()

    df["signal_raw"] = 0
    df.loc[df["ma_short"] > df["ma_long"], "signal_raw"] = 1
    df.loc[df["ma_short"] < df["ma_long"], "signal_raw"] = -1

    df["signal"] = df["signal_raw"].replace(0, pd.NA).ffill().fillna(0).astype(float)

    return df
