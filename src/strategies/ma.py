import pandas as pd

def ma_strategy(
    df: pd.DataFrame,
    short_window: int = 10,
    long_window: int = 50,
    price_col: str = "Close"
) -> pd.DataFrame:
    """
    MA Crossover 策略（带持仓延续）
    signal_raw：
        1  = 多头
        -1 = 空头
        0  = 无明确信号
    signal：
        使用 ffill 延续上一根持仓方向
    """

    if price_col not in df.columns:
        raise ValueError(f"MA strategy: df 必须包含列 '{price_col}'")

    df = df.copy()
    price = df[price_col]

    df["ma_short"] = price.rolling(short_window).mean()
    df["ma_long"]  = price.rolling(long_window).mean()

    # 补全前期均线，避免前段出现 NaN
    df[["ma_short", "ma_long"]] = df[["ma_short", "ma_long"]].bfill()

    # 原始信号（当前K方向判断）
    df["signal_raw"] = 0
    df.loc[df["ma_short"] > df["ma_long"], "signal_raw"] = 1
    df.loc[df["ma_short"] < df["ma_long"], "signal_raw"] = -1

    # 持仓延续（专业做法）
    df["signal"] = df["signal_raw"].replace(0, pd.NA).ffill().fillna(0).astype(float)

    return df
