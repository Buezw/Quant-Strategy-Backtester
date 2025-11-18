import pandas as pd

def rsi_strategy(
    df: pd.DataFrame,
    window: int = 14,
    rsi_low: int = 30,
    rsi_high: int = 70,
    price_col: str = "Close"
) -> pd.DataFrame:
    """
    RSI Mean Reversion（带持仓延续）
    - RSI < low  → 多头
    - RSI > high → 空头
    """

    df = df.copy()
    price = df[price_col]

    delta = price.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - 100 / (1 + rs)

    # raw signal
    df["signal_raw"] = 0
    df.loc[df["rsi"] < rsi_low, "signal_raw"] = 1
    df.loc[df["rsi"] > rsi_high, "signal_raw"] = -1

    # 持仓延续
    df["signal"] = df["signal_raw"].replace(0, pd.NA).ffill().fillna(0).astype(float)

    return df
