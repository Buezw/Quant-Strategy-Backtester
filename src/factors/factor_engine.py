import pandas as pd
from .technical import add_technical_factors
from .volatility import add_vol_factors
from .volume import add_volume_factors
from .stats import add_stat_factors

def generate_factors(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df = add_technical_factors(df)
    df = add_vol_factors(df)
    df = add_volume_factors(df)
    df = add_stat_factors(df)

    df = df.dropna()
    return df
