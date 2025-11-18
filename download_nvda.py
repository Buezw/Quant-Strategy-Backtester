import yfinance as yf
import pandas as pd
import os

def download_ohlcv(
    symbol="NVDA",
    period="700d",
    interval="60m",
    output_path="data/raw/nvda_1h_700d.csv"
):
    print(f"📡 Downloading {symbol} {interval} OHLCV data from Yahoo Finance...")

    # 下载数据
    df = yf.download(symbol, period=period, interval=interval)

    if df.empty:
        raise ValueError("❌ ERROR: No data returned — check network or Yahoo API limits.")

    # 确保包含 OHLCV
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[ohlcv_cols].reset_index()

    # 处理缺失值
    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)

    # 创建文件夹
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存数据
    df.to_csv(output_path, index=False)

    print(f"✅ Saved {symbol} OHLCV data to: {output_path}")
    print(df.head())

    return df


if __name__ == "__main__":
    df = download_ohlcv()
