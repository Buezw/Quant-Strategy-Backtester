def ma_crossover(df, short=10, long=50):
    df = df.copy()
    df["ma_short"] = df["Close"].rolling(short).mean()
    df["ma_long"] = df["Close"].rolling(long).mean()

    df["signal"] = 0
    df.loc[df["ma_short"] > df["ma_long"], "signal"] = 1
    df.loc[df["ma_short"] < df["ma_long"], "signal"] = -1
    return df
