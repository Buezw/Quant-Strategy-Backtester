# src/plot/equity.py
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use("seaborn-v0_8-darkgrid")


def _ensure_dir(filepath):
    folder = os.path.dirname(filepath)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)


def plot_equity_and_drawdown(df: pd.DataFrame, save_path: str):
    """
    绘制净值曲线 + 回撤曲线。
    要求 df 中包含:
    - equity
    """

    if "equity" not in df.columns:
        raise ValueError("df must contain 'equity' column")

    _ensure_dir(save_path)

    equity = df["equity"]
    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})

    # --- Equity curve ---
    ax[0].plot(equity, label="Equity", linewidth=1.6)
    ax[0].set_title("Equity Curve")
    ax[0].legend()

    # --- Drawdown curve ---
    ax[1].fill_between(df.index, drawdown, 0, color="red", alpha=0.4)
    ax[1].set_title("Drawdown (%)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"📁 Saved equity/drawdown chart: {save_path}")
