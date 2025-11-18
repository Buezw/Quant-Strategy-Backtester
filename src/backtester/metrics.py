import numpy as np
import pandas as pd


# ------------------------------
# Helper: 清洗收益数据
# ------------------------------
def _clean_returns(ret):
    """
    接受一个 pandas Series:
    - 去掉 NaN
    - 去掉 inf
    - 必须是 float
    """
    ret = pd.Series(ret).astype(float)
    ret = ret.replace([np.inf, -np.inf], np.nan).dropna()

    if len(ret) == 0:
        return pd.Series([0.0])

    return ret


# ------------------------------
# 年化因子（按频率）
# ------------------------------
def _annualize_factor(freq="1d"):
    freq = str(freq).lower()

    if freq == "1d":
        return 252
    if freq == "1h":
        return 252 * 6.5     # 美股交易 6.5 hours/day
    if freq == "1min":
        return 252 * 6.5 * 60

    raise ValueError(f"Unsupported freq '{freq}', valid options: 1d, 1h, 1min")


# ------------------------------
# Sharpe Ratio
# ------------------------------
def sharpe_ratio(df_or_returns, freq="1d"):
    """
    输入可以是：
    - df: 包含 net_ret
    - pandas Series: 直接是收益

    计算：
        Sharpe = mean(ret) / std(ret) * sqrt(annual_factor)
    """
    if isinstance(df_or_returns, pd.DataFrame):
        if "net_ret" not in df_or_returns.columns:
            raise ValueError("DataFrame 需要包含 net_ret 列")
        returns = df_or_returns["net_ret"]
    else:
        returns = df_or_returns

    returns = _clean_returns(returns)
    ann = _annualize_factor(freq)

    if returns.std() == 0:
        return 0.0

    return returns.mean() / returns.std() * np.sqrt(ann)


# ------------------------------
# Max Drawdown
# ------------------------------
def max_drawdown(df):
    """
    计算基于 equity 的最大回撤。
    """
    if "equity" not in df.columns:
        raise ValueError("DataFrame 需要包含 equity 列")

    equity = df["equity"].astype(float)

    running_max = equity.cummax()
    dd = (equity - running_max) / running_max

    return dd.min()    # negative number


# ------------------------------
# Volatility（年化波动率）
# ------------------------------
def volatility(df_or_returns, freq="1d"):
    if isinstance(df_or_returns, pd.DataFrame):
        if "net_ret" not in df_or_returns.columns:
            raise ValueError("DataFrame 需要包含 net_ret 列")
        returns = df_or_returns["net_ret"]
    else:
        returns = df_or_returns

    returns = _clean_returns(returns)
    ann = _annualize_factor(freq)

    return returns.std() * np.sqrt(ann)
