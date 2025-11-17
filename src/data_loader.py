import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    # ---- 清理列名 ----
    df.columns = [c.strip() for c in df.columns]

    # ---- 确保 Datetime 列存在 ----
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        df.rename(columns={"Datetime": "Date"}, inplace=True)
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        raise ValueError("No Date/Datetime column found in CSV.")

    # ---- 选择价格列 ----
    price_col = None
    for col in df.columns:
        if col.lower() in ["close", "price", "nvda"]:
            price_col = col
            break

    if price_col is None:
        raise ValueError("No price column found. Expected Close/NVDA.")

    # ---- 强制将价格列转 float ----
    df["Close"] = pd.to_numeric(df[price_col], errors="coerce")

    # ---- 最终保留两列 ----
    df = df[["Date", "Close"]].dropna().reset_index(drop=True)

    return df
