import pandas as pd
from src.data_loader import load_data
from src.strategy_ma import ma_crossover
from src.backtester.engine import backtest

def main():
    print("=== Loading Raw Data ===")
    df_raw = load_data("data/raw/data.csv")
    print(df_raw.dtypes)
    print(df_raw.head(), "\n")

    print("=== After Strategy (ma_crossover) ===")
    df_strategy = ma_crossover(df_raw.copy(), short=10, long=50)
    print(df_strategy.dtypes)
    print(df_strategy.head(), "\n")

    print("=== After Backtest ===")
    df_bt = backtest(df_strategy.copy(), initial_capital=10000)
    print(df_bt.dtypes)
    print(df_bt.head(), "\n")

    # 特别检查收益列有没有 Timestamp
    print("=== Column type check ===")
    for col in ["price_ret", "strategy_ret", "net_ret", "equity"]:
        if col in df_bt.columns:
            print(f"{col}: {df_bt[col].dtype}")
        else:
            print(f"{col}: NOT FOUND")

if __name__ == "__main__":
    main()
