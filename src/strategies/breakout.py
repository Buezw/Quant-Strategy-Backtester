import pandas as pd

def breakout_strategy(
    df: pd.DataFrame,
    high_window: int = 20,
    low_window: int = 10
) -> pd.DataFrame:
    """
    海龟突破策略（Donchian Channel Breakout）
    - Close > past N-day high → long
    - Close < past M-day low  → short
    """

    df = df.copy()

    df["highest"] = df["High"].rolling(high_window).max()
    df["lowest"]  = df["Low"].rolling(low_window).min()

    # 避免前期 NaN
    df[["highest", "lowest"]] = df[["highest", "lowest"]].bfill()

    # 原始信号
    df["signal_raw"] = 0
    df.loc[df["Close"] > df["highest"], "signal_raw"] = 1
    df.loc[df["Close"] < df["lowest"],  "signal_raw"] = -1

    # 持仓延续
    df["signal"] = df["signal_raw"].replace(0, pd.NA).ffill().fillna(0).astype(float)

    return df
