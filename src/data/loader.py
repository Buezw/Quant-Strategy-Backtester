import pandas as pd

def load_data(path: str):
    """
    加载 OHLCV 数据（从 CSV），自动处理以下问题：
    - Yahoo Finance 产生的第二行垃圾表头
    - 字符串数字 / 空值 / 错列
    - 强制 OHLCV 数值化
    - 修复 Volume 导致 rolling corr 报错
    - 设置 timestamp index
    """
    # ---- 读取 CSV ----
    df = pd.read_csv(path)

    # ---- 1. 删除 Yahoo 第二表头（例如 ",NVDA,NVDA,...") ----
    # 检查第一行是否是日期，不是的话说明是垃圾 header → 删除
    try:
        pd.to_datetime(df.iloc[0, 0])
    except:
        df = df[1:].reset_index(drop=True)

    # ---- 2. 重命名时间列 ----
    # CSV 可能有 Datetime / Date / timestamp
    for time_col in ["Datetime", "datetime", "Timestamp", "timestamp", "Date"]:
        if time_col in df.columns:
            df.rename(columns={time_col: "timestamp"}, inplace=True)
            break

    if "timestamp" not in df.columns:
        raise ValueError("❌ CSV 缺少 timestamp 列")

    # ---- 3. 转换时间格式 ----
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # ---- 4. 强制 OHLCV 数值化（核心增强）----
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]

    for col in numeric_cols:
        if col not in df.columns:
            # 如果缺某列 → 自动补 0（避免因子崩溃）
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- 5. 修复异常 Volume（rolling corr 经常报错）----
    df["Volume"] = df["Volume"].fillna(0.0)

    # ---- 6. 删除全空行 ----
    df = df.dropna(subset=["Close"])

    # ---- 7. 按时间排序 ----
    df = df.sort_values("timestamp")

    # ---- 8. 设置时间 index ----
    df = df.set_index("timestamp")

    return df
