import pandas as pd

def generate_trade_log(df_bt: pd.DataFrame):

    trades = []

    df = df_bt.copy()
    df = df.reset_index()       
    df.rename(columns={"index": "timestamp"}, inplace=True)

    prev_pos = 0

    for i in range(len(df)):
        row = df.iloc[i]
        pos = row["position"]

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
