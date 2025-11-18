import pandas as pd

def load_data(path: str):
    df = pd.read_csv(path)

    try:
        pd.to_datetime(df.iloc[0, 0])
    except:
        df = df[1:].reset_index(drop=True)

    for time_col in ["Datetime", "datetime", "Timestamp", "timestamp", "Date"]:
        if time_col in df.columns:
            df.rename(columns={time_col: "timestamp"}, inplace=True)
            break

    if "timestamp" not in df.columns:
        raise ValueError("CSV need timestamp ")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Volume"] = df["Volume"].fillna(0.0)
    df = df.dropna(subset=["Close"])
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")

    return df
