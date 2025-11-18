import pandas as pd
from .technical import add_technical_factors
from .volatility import add_vol_factors
from .volume import add_volume_factors
from .stats import add_stat_factors

def generate_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：原始行情数据（OHLCV）
    输出：添加了多种 Alpha 因子的 DataFrame
    因子包括：
        - 技术因子（MA, RSI, MACD, MOM）
        - 波动率因子（Vol, ATR）
        - 成交量因子（VolumeSpike）
        - 统计因子（Z-score, rolling_corr）
    """

    df = df.copy()

    df = add_technical_factors(df)
    df = add_vol_factors(df)
    df = add_volume_factors(df)
    df = add_stat_factors(df)

    df = df.dropna()
    return df
