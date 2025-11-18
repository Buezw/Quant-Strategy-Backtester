import pandas as pd
import numpy as np

def add_stat_factors(df, rolling: int = 20):
    """
    添加统计类因子（相关性、波动等）
    自动处理 Volume 无效 / 全零 / NaN 的问题
    """

    # ---- 修复 Volume，确保有效 ----
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)

    # 如果 Volume 全部相同（0 或常数），rolling corr 会崩 → 我们提供 fallback
    if df["Volume"].nunique() <= 1:
        df["roll_corr"] = 0.0
    else:
        # 安全 try-catch，任何异常 fallback = 0
        try:
            df["roll_corr"] = (
                df["Close"]
                .rolling(rolling)
                .corr(df["Volume"].rolling(rolling).mean())
            )
        except Exception:
            df["roll_corr"] = 0.0

    # ---- 其他统计因子 ----
    df["vol_mean"] = df["Close"].rolling(rolling).std()
    df["vol_ratio"] = df["Close"] / df["Close"].rolling(rolling).mean()

    # 避免 NaN
    df = df.fillna(0)

    return df
