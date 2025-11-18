def add_volume_factors(df):
    df["vol_spike"] = df["Volume"] / df["Volume"].rolling(20).mean()
    return df
