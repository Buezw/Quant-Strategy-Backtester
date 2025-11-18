import pandas as pd
import numpy as np
from src.backtester.engine import run_backtest
from src.backtester.metrics import sharpe_ratio
from src.strategies.ma import ma_strategy

import matplotlib.pyplot as plt
import seaborn as sns
import os


def _ensure_dir(path: str):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)


def grid_search_ma(
    df_raw: pd.DataFrame,
    short_range=[5, 10, 20],
    long_range=[50, 100, 150],
    commission=0.0005,
    slippage=0.0002,
    save_path: str = None,
):
    """
    对 MA 策略进行参数优化。

    参数：
    - df_raw: 原始数据（未加入 signal/no returns）
    - short_range: 短均线搜索范围
    - long_range: 长均线搜索范围
    - commission/slippage: 交易成本
    - save_path: 如果提供则自动保存 heatmap

    返回：
    - best_param: {"short": x, "long": y, "sharpe": z}
    - result_df: 所有参数组合的结果 DataFrame
    """

    results = []

    for short in short_range:
        for long in long_range:

            if short >= long:
                continue   # MA 策略不能 short>=long，否则没意义

            # 1) 生成 signal
            df_sig = ma_strategy(df_raw, short=short, long=long)

            # 2) 回测
            df_bt = run_backtest(df_sig, commission=commission, slippage=slippage)

            # 3) 计算 sharpe
            sharpe = sharpe_ratio(df_bt)

            results.append({
                "short": short,
                "long": long,
                "sharpe": sharpe
            })

    # 转为 DataFrame
    res_df = pd.DataFrame(results).sort_values("sharpe", ascending=False).reset_index(drop=True)

    # 最优参数
    best = res_df.iloc[0].to_dict()

    # -------------------------
    # (Optional) 生成 Heatmap
    # -------------------------
    if save_path:
        _ensure_dir(save_path)

        pivot = res_df.pivot(index="short", columns="long", values="sharpe")

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            pivot,
            annot=True,
            cmap="viridis",
            fmt=".3f",
            cbar_kws={"label": "Sharpe Ratio"},
        )
        plt.title("MA Parameter Grid Search (Sharpe)")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📁 Saved heatmap to: {save_path}")

    return best, res_df
