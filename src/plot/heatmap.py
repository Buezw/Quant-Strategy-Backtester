# src/plot/heatmap.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def _ensure_dir(filepath):
    folder = os.path.dirname(filepath)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)


def plot_heatmap(df: pd.DataFrame, save_path: str, metric="sharpe"):
    """
    根据 gridsearch 结果绘制 heatmap。
    df 必须包含 columns: short, long, metric
    """

    _ensure_dir(save_path)

    if metric not in df.columns:
        raise ValueError(f"df must contain {metric}")

    pivot = df.pivot(index="short", columns="long", values=metric)

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".3f",
                cbar_kws={"label": metric})

    plt.title(f"Grid Search Heatmap ({metric})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"📁 Saved heatmap chart: {save_path}")
