import numpy as np

def compute_trade_stats(trades):
    """
    输入：
        trades : trade_log 的 list(dict)
    
    输出：
        stats : dict，包含 win rate、profit factor 等
    """

    if len(trades) == 0:
        return {
            "total_trades": 0,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "max_consecutive_losses": np.nan,
            "long_trades": 0,
            "short_trades": 0,
            "long_win_rate": np.nan,
            "short_win_rate": np.nan,
        }

    # 提取所有平仓类交易（有 pnl 的）
    closed = [t for t in trades if t["pnl"] != 0]

    pnls = np.array([t["pnl"] for t in closed])

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    # 计算总交易数（含开仓）
    total_trades = len(closed)

    # 胜率
    win_rate = len(wins) / total_trades if total_trades > 0 else np.nan

    # 利润因子（Profit Factor）
    profit_factor = (wins.sum() / abs(losses.sum())) if len(losses) > 0 else np.nan

    # 平均盈亏
    avg_win = wins.mean() if len(wins) > 0 else np.nan
    avg_loss = losses.mean() if len(losses) > 0 else np.nan

    # 最大连续亏损
    max_consecutive_losses = 0
    current = 0
    for pnl in pnls:
        if pnl < 0:
            current += 1
            max_consecutive_losses = max(max_consecutive_losses, current)
        else:
            current = 0

    # 多头/空头分析
    long_trades = [t for t in closed if t["position"] == 1]
    short_trades = [t for t in closed if t["position"] == -1]

    long_pnls = np.array([t["pnl"] for t in long_trades])
    short_pnls = np.array([t["pnl"] for t in short_trades])

    long_win_rate = (long_pnls > 0).mean() if len(long_trades) > 0 else np.nan
    short_win_rate = (short_pnls > 0).mean() if len(short_trades) > 0 else np.nan

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_consecutive_losses": max_consecutive_losses,
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "long_win_rate": long_win_rate,
        "short_win_rate": short_win_rate,
    }
