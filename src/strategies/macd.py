import pandas as pd

def macd_strategy(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_window: int = 9,
    price_col: str = "Close"
) -> pd.DataFrame:
    """
    MACD 趋势跟随（带持仓延续）
    - DIF 上穿 DEA → 多头
    - DIF 下穿 DEA → 空头
    """

    df = df.copy()
    price = df[price_col]

    df["ema_fast"] = price.ewm(span=fast, adjust=False).mean()
    df["ema_slow"]  = price.ewm(span=slow, adjust=False).mean()

    df["macd"] = df["ema_fast"] - df["ema_slow"]
    df["macd_signal"] = df["macd"].ewm(span=signal_window, adjust=False).mean()

    # 原始信号
    df["signal_raw"] = 0
    df.loc[df["macd"] > df["macd_signal"], "signal_raw"] = 1
    df.loc[df["macd"] < df["macd_signal"], "signal_raw"] = -1

    # 持仓延续
    df["signal"] = df["signal_raw"].replace(0, pd.NA).ffill().fillna(0).astype(float)

    return df
