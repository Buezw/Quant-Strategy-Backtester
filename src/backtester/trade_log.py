import pandas as pd

def generate_trade_log(df_bt: pd.DataFrame):
    """
    生成交易日志：
    - 每当 position 变化时记录交易
    - 自动适配 timestamp index（无 Date 列）
    """

    trades = []

    # 确保 index 是时间戳
    df = df_bt.copy()
    df = df.reset_index()        # timestamp → 列
    df.rename(columns={"index": "timestamp"}, inplace=True)

    prev_pos = 0

    for i in range(len(df)):
        row = df.iloc[i]
        pos = row["position"]

        # 仓位改变 → 交易行为
        if pos != prev_pos:
            if pos > prev_pos:
                action = "BUY"
            elif pos < prev_pos:
                action = "SELL"
            else:
                action = "HOLD"

            trades.append({
                "timestamp": row["timestamp"],
                "action": action,
                "price": row["Close"],
                "prev_pos": prev_pos,
                "new_pos": pos,
                "pnl": row.get("strategy_ret", 0),
            })

        prev_pos = pos

    return trades
