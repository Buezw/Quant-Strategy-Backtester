import pandas as pd

from .ma import ma_strategy
from .breakout import breakout_strategy
from .momentum import momentum_strategy
from .zscore import zscore_strategy


def meta_regime_strategy(
    df: pd.DataFrame,
    # —— Regime 判定用的均线参数 ——
    trend_ma_short: int = 50,
    trend_ma_long: int = 200,
    trend_threshold: float = 0.01,   # |MA_short - MA_long| / Close > 1% 视为趋势

    # —— 趋势期使用的底层策略选择 & 参数 ——
    trend_mode: str = "momentum",    # 可选："ma" / "breakout" / "momentum"
    ma_short_window: int = 20,
    ma_long_window: int = 100,
    breakout_high_window: int = 20,
    breakout_low_window: int = 10,
    momentum_lookback: int = 20,

    # —— 震荡期底层策略（这里固定用 Z-score 均值回复） ——
    zscore_window: int = 20,
    zscore_entry: float = 2.0,
) -> pd.DataFrame:
    """
    Regime Switching Meta-Strategy

    1. 用长短期 MA 差值判断市场：
       - |ma_short - ma_long| / Close > trend_threshold → trend regime
       - 否则 → range regime

    2. trend regime:
       - 用 trend_mode 对应的底层策略：ma / breakout / momentum

    3. range regime:
       - 用 zscore 均值回复策略

    最终输出:
      - df["regime"]     ∈ {"trend", "range"}
      - df["signal_trend"]
      - df["signal_range"]
      - df["signal"]     （根据 regime 选择其一，带持仓延续）
    """

    df = df.copy()

    # ================
    # 1) 计算趋势判定用的 MA
    # ================
    price = df["Close"]

    df["trend_ma_short"] = price.rolling(trend_ma_short).mean()
    df["trend_ma_long"] = price.rolling(trend_ma_long).mean()
    df[["trend_ma_short", "trend_ma_long"]] = df[
        ["trend_ma_short", "trend_ma_long"]
    ].bfill()

    # MA 差值强度
    ma_diff_strength = (df["trend_ma_short"] - df["trend_ma_long"]).abs() / price
    df["ma_diff_strength"] = ma_diff_strength

    # 定义 regime：True = trend, False = range
    df["is_trend"] = df["ma_diff_strength"] > trend_threshold
    df["regime"] = df["is_trend"].map({True: "trend", False: "range"})

    # ================
    # 2) 计算各底层策略信号
    # ================

    # 趋势期策略
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
        raise ValueError(f"未知 trend_mode: {trend_mode}, 只能是 'ma' / 'breakout' / 'momentum'")

    # 震荡期策略：Z-score 均值回复
    df_range = zscore_strategy(
        df.copy(),
        window=zscore_window,
        z_entry=zscore_entry,
    )

    df["signal_trend"] = df_trend["signal"]
    df["signal_range"] = df_range["signal"]

    # ================
    # 3) 按 regime 选择对应 signal
    # ================
    df["signal_raw"] = 0.0
    df.loc[df["is_trend"], "signal_raw"] = df.loc[df["is_trend"], "signal_trend"]
    df.loc[~df["is_trend"], "signal_raw"] = df.loc[~df["is_trend"], "signal_range"]

    # 再做一次持仓延续，避免 regime 切换时出现 0
    df["signal"] = (
        df["signal_raw"].replace(0, pd.NA).ffill().fillna(0).astype(float)
    )

    return df
