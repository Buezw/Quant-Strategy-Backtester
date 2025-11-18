import numpy as np
import pandas as pd


# Cleaning Function
def _clean_returns(ret):

    ret = pd.Series(ret).astype(float)
    ret = ret.replace([np.inf, -np.inf], np.nan).dropna()

    if len(ret) == 0:
        return pd.Series([0.0])

    return ret


# Return Rate
def _annualize_factor(freq="1d"):
    freq = str(freq).lower()

    if freq == "1d":
        return 252
    if freq == "1h":
        return 252 * 6.5    
    if freq == "1min":
        return 252 * 6.5 * 60

    raise ValueError(f"Unsupported freq '{freq}', valid options: 1d, 1h, 1min")


# Sharpe Ratio
def sharpe_ratio(df_or_returns, freq="1d"):
    if isinstance(df_or_returns, pd.DataFrame):
        if "net_ret" not in df_or_returns.columns:
            raise ValueError("DataFrame need net_ret")
        returns = df_or_returns["net_ret"]
    else:
        returns = df_or_returns

    returns = _clean_returns(returns)
    ann = _annualize_factor(freq)

    if returns.std() == 0:
        return 0.0

    return returns.mean() / returns.std() * np.sqrt(ann)


# Max Drawdown
def max_drawdown(df):
    if "equity" not in df.columns:
        raise ValueError("DataFrame need equity col")

    equity = df["equity"].astype(float)

    running_max = equity.cummax()
    dd = (equity - running_max) / running_max

    return dd.min()    # negative number


# Volatility
def volatility(df_or_returns, freq="1d"):
    if isinstance(df_or_returns, pd.DataFrame):
        if "net_ret" not in df_or_returns.columns:
            raise ValueError("DataFrame need net_ret")
        returns = df_or_returns["net_ret"]
    else:
        returns = df_or_returns

    returns = _clean_returns(returns)
    ann = _annualize_factor(freq)

    return returns.std() * np.sqrt(ann)
