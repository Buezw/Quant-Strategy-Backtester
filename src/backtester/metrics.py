import numpy as np
import pandas as pd


def _clean_returns(series):
    """确保 returns 列只有 float，不含 Timestamp/object"""
    s = pd.to_numeric(series, errors="coerce")  # 强制 float
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.dropna()

    return s


def sharpe_ratio(df):
    returns = _clean_returns(df["net_ret"])

    if len(returns) < 2:
        return 0.0

    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0

    return np.sqrt(252) * returns.mean() / std


def max_drawdown(df):
    equity = pd.to_numeric(df["equity"], errors="coerce").dropna()

    if len(equity) < 2:
        return 0.0

    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return drawdown.min()


def volatility(df):
    returns = _clean_returns(df["net_ret"])

    if len(returns) < 2:
        return 0.0

    return np.sqrt(252) * returns.std()
