def add_vol_factors(df):
    df["vol20"] = df["ret1"].rolling(20).std()
    df["atr"] = (df["High"] - df["Low"]).rolling(14).mean()
    return df
