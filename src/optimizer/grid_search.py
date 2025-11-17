import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from src.strategy_ma import ma_crossover

from src.backtester.engine import backtest
from src.backtester.metrics import sharpe_ratio


def grid_search_ma(
    df,
    short_range=[5, 10, 20],
    long_range=[30, 50, 100],
    commission=0.0005,
    slippage=0.0002,
    save_path="results/charts/heatmap_sharpe.png"
):

    results = []

    for short in short_range:
        for long in long_range:

            if short >= long:
                continue  # MA 策略里短均线必须 < 长均线
            
            df_tmp = df.copy()

            # 策略
            df_tmp = ma_crossover(df_tmp, short=short, long=long)

            # 回测（带手续费和滑点）
            df_tmp = backtest(df_tmp, commission=commission, slippage=slippage)
            df_tmp = df_tmp.dropna()

            # 策略收益
            ret = df_tmp["net_ret"].dropna()


            sharpe = sharpe_ratio(df_tmp)
    
            results.append([short, long, sharpe])

    # 整理为 DataFrame
    res_df = pd.DataFrame(results, columns=["short", "long", "sharpe"])

    # —— 构造热力图矩阵 ——
    heatmap_data = res_df.pivot(index="long", columns="short", values="sharpe")

    # —— 保存热力图 ——
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.switch_backend("Agg")

    plt.figure(figsize=(10, 6))
    plt.title("Sharpe Ratio Heatmap (MA Strategy)")
    plt.xlabel("Short MA")
    plt.ylabel("Long MA")

    plt.imshow(heatmap_data, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Sharpe Ratio")

    # 坐标轴标签
    plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
    plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"📁 Saved heatmap to: {save_path}")

    # —— 找 Sharpe 最大的参数组合 ——
    best = res_df.loc[res_df["sharpe"].idxmax()].to_dict()

    return best, res_df
