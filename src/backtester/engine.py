import pandas as pd
import numpy as np


def backtest(df, initial_capital=10000, commission=0.0005, slippage=0.0002):
    """
    Vectorized backtest with:
    - signal: 1 long, -1 short, 0 flat
    - net_ret: 含手续费和滑点之后的净收益
    - equity: 净值曲线
    """

    df = df.copy()

    # 确保 signal 存在
    df["signal"] = df["signal"].fillna(0)

    # 持仓（上一根 K 的信号）
    df["position"] = df["signal"].shift(1).fillna(0)

    # 价格收益
    df["price_ret"] = df["Close"].pct_change().fillna(0)

    # 策略收益（不含成本）
    df["strategy_ret"] = df["position"] * df["price_ret"]

    # 换仓标记（只有 signal 变化时才收费）
    df["trade_flag"] = df["signal"].diff().fillna(0).abs()

    # 手续费 & 滑点
    df["commission_cost"] = df["trade_flag"] * commission
    df["slippage_cost"] = df["trade_flag"] * slippage

    # 合并成本
    df["cost"] = df["commission_cost"] + df["slippage_cost"]

    # 最终收益：策略收益 – 成本
    df["net_ret"] = df["strategy_ret"] - df["cost"]

    # 净值曲线
    df["equity"] = initial_capital * (1 + df["net_ret"]).cumprod()

    return df
