import matplotlib.pyplot as plt
import numpy as np
import os


# ===========================================
# 1. 净值曲线 + 回撤曲线
# ===========================================
def plot_equity_and_drawdown(df, save_path="results/charts/equity_drawdown.png"):
    """
    绘制：
      - Equity Curve
      - Drawdown Curve
    并保存为 PNG（服务器用 Agg 后端）
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.switch_backend("Agg")  # 服务器环境需要

    equity = df["equity"]
    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Equity
    ax1.plot(df["Date"], equity, color="blue", label="Equity")
    ax1.set_title("Equity Curve")
    ax1.legend()
    ax1.grid(True)

    # Drawdown
    ax2.plot(df["Date"], drawdown, color="red", label="Drawdown")
    ax2.fill_between(df["Date"], drawdown, 0, color="red", alpha=0.2)
    ax2.set_title("Drawdown Curve")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"📁 Saved equity & drawdown chart to: {save_path}")


# ===========================================
# 2. 价格图 + 买卖点标注 + 持仓背景
# ===========================================
def plot_entry_exit(df, trades, save_path="results/charts/entry_exit.png"):
    """
    绘制：
      - Price curve
      - Buy / Sell / Reverse markers
      - Long/Short shading
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.switch_backend("Agg")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Price
    ax.plot(df["Date"], df["Close"], color="blue", label="Price", linewidth=1.2)

    # buy/sell/reverse markers
    buy_x, buy_y = [], []
    sell_x, sell_y = [], []
    reverse_x, reverse_y = [], []

    for t in trades:
        time = t["time"]
        price = t["price"]

        if t["action"] == "BUY":
            buy_x.append(time)
            buy_y.append(price)

        elif t["action"] == "SELL":
            sell_x.append(time)
            sell_y.append(price)

        elif t["action"] == "REVERSE_OPEN":
            if t["position"] == 1:
                buy_x.append(time)
                buy_y.append(price)
            else:
                sell_x.append(time)
                sell_y.append(price)

        elif t["action"] == "REVERSE_CLOSE":
            reverse_x.append(time)
            reverse_y.append(price)

    # 🎯 画点
    ax.scatter(buy_x, buy_y, color="green", s=40, label="BUY")
    ax.scatter(sell_x, sell_y, color="red", s=40, label="SELL")
    ax.scatter(reverse_x, reverse_y, color="yellow", s=60, marker="x", label="REVERSE")

    # 🎯 持仓背景（long/short）
    pos = df["signal"].replace({1: "long", -1: "short", 0: "flat"})

    for i in range(1, len(df)):
        if pos[i] == "long":
            ax.axvspan(df["Date"][i - 1], df["Date"][i], color="green", alpha=0.03)
        elif pos[i] == "short":
            ax.axvspan(df["Date"][i - 1], df["Date"][i], color="red", alpha=0.03)

    ax.set_title("Entry / Exit Chart")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"📁 Saved entry/exit chart to: {save_path}")
