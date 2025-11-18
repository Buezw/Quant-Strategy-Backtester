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
    results = []

    for short in short_range:
        for long in long_range:
            if short >= long:
                continue

            df_sig = ma_strategy(df_raw, short=short, long=long)
            df_bt = run_backtest(df_sig, commission=commission, slippage=slippage)
            sharpe = sharpe_ratio(df_bt)

            results.append({
                "short": short,
                "long": long,
                "sharpe": sharpe
            })

    res_df = pd.DataFrame(results).sort_values("sharpe", ascending=False).reset_index(drop=True)
    best = res_df.iloc[0].to_dict()

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

        print(f"Saved heatmap to: {save_path}")

    return best, res_df
