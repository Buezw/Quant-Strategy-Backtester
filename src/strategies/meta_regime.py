import pandas as pd

from .ma import ma_strategy
from .breakout import breakout_strategy
from .momentum import momentum_strategy
from .zscore import zscore_strategy


def meta_regime_strategy(
    df: pd.DataFrame,
    # —— Regime 判定 ——
    trend_ma_short: int = 50,
    trend_ma_long: int = 200,
    trend_threshold: float = 0.01,   # |MA_short - MA_long| / Close > 1% → trend

    # —— 趋势期策略参数 ——
    trend_mode: str = "momentum",    # 可选："ma" / "breakout" / "momentum"
    ma_short_window: int = 20,
    ma_long_window: int = 100,
    breakout_high_window: int = 20,
    breakout_low_window: int = 10,
    momentum_lookback: int = 20,

    # —— 震荡期策略参数 ——
    zscore_window: int = 20,
    zscore_entry: float = 2.0,
):
    """
    Regime-Switching Meta Strategy（非 ML）
    根据行情状态自动切换策略：

    - Trend Regime -> 使用 trend_mode 对应策略
    - Range Regime -> 使用 Z-score 均值回复策略

    输出：
        df["regime"]
        df["signal_trend"]
        df["signal_range"]
        df["signal"]
    """

    df = df.copy()
    price = df["Close"]

    # ===========================
    # 1) 用 MA 差值判断市场状态
    # ===========================
    df["trend_ma_short"] = price.rolling(trend_ma_short).mean()
    df["trend_ma_long"] = price.rolling(trend_ma_long).mean()
    df[["trend_ma_short", "trend_ma_long"]] = df[
        ["trend_ma_short", "trend_ma_long"]
    ].bfill()

    df["ma_diff_strength"] = (
        (df["trend_ma_short"] - df["trend_ma_long"]).abs() / price
    )

    df["is_trend"] = df["ma_diff_strength"] > trend_threshold
    df["regime"] = df["is_trend"].map({True: "trend", False: "range"})

    # ===========================
    # 2) 计算趋势期策略信号
    # ===========================
    if trend_mode == "ma":
        df_trend = ma_strategy(
            df.copy(),
            short_window=ma_short_window,
            long_window=ma_long_window,
        )

    elif trend_mode == "breakout":
        df_trend = breakout_strategy(
            df.copy(),
            high_window=breakout_high_window,
            low_window=breakout_low_window,
        )

    elif trend_mode == "momentum":
        df_trend = momentum_strategy(
            df.copy(),
            lookback=momentum_lookback,
        )

    else:
        raise ValueError(
            f"未知 trend_mode: {trend_mode}, 只能是 'ma' / 'breakout' / 'momentum'"
        )

    df["signal_trend"] = df_trend["signal"]

    # ===========================
    # 3) 计算震荡期均值回复信号
    # ===========================
    df_range = zscore_strategy(
        df.copy(),
        window=zscore_window,
        z_entry=zscore_entry,
    )
    df["signal_range"] = df_range["signal"]

    # ===========================
    # 4) 最终信号按 Regime 切换
    # ===========================
    df["signal_raw"] = 0.0
    df.loc[df["is_trend"], "signal_raw"] = df.loc[df["is_trend"], "signal_trend"]
    df.loc[~df["is_trend"], "signal_raw"] = df.loc[~df["is_trend"], "signal_range"]

    # 避免 regime 切换产生 0：向前填充
    df["signal"] = (
        df["signal_raw"]
        .replace(0, pd.NA)
        .ffill()
        .fillna(0)
        .astype(float)
    )

    return df
