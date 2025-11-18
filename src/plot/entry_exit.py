# src/plot/entry_exit.py
import os
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("seaborn-v0_8-darkgrid")


def _ensure_dir(filepath):
    folder = os.path.dirname(filepath)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)


def plot_entry_exit(df: pd.DataFrame, save_path: str):
    _ensure_dir(save_path)

    if "Close" not in df.columns:
        raise ValueError("df must contain Close")
    if "position" not in df.columns:
        raise ValueError("df must contain position")

    price = df["Close"]
    pos = df["position"].fillna(0)

    buy_signal = pos.diff() == 1
    sell_signal = pos.diff() == -1

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(price, label="Price", linewidth=1.3)

    ax.scatter(df.index[buy_signal], price[buy_signal],
               marker="^", color="green", s=60, label="BUY")

    ax.scatter(df.index[sell_signal], price[sell_signal],
               marker="v", color="red", s=60, label="SELL")

    ax.set_title("Entry / Exit Plot")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"📁 Saved entry/exit chart: {save_path}")
