import yfinance as yf

# 下载最近 60 天 / 1 小时级别（60m）NVDA 数据
df = yf.download(
    "NVDA",
    period="700d",      # 只能取最近 60 天
    interval="60m"     # 60 分钟粒度
)

# 如果没有数据（网络错误 / Yahoo 限制），df 会是空的
if df.empty:
    raise ValueError("No data returned — check your network or Yahoo limits.")

# 只保留收盘价
df = df[["Close"]].reset_index()

# 保存 CSV
output_path = "data/raw/nvda_1h_60d.csv"
df.to_csv(output_path, index=False)

print(f"Saved NVDA 1H data to {output_path}")
print(df.head())
