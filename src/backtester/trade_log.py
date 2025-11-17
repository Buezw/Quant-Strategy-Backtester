def generate_trade_log(df):
    """
    输入：
        df: 包含 Date, Close, signal 的 DataFrame
    
    输出：
        trades: list(dict)
        
    每条记录包含：
        time, action, price, position, pnl
    """
    trades = []
    position = 0
    entry_price = None

    for i in range(1, len(df)):
        prev_sig = df.loc[i-1, "signal"]
        curr_sig = df.loc[i, "signal"]
        price = df.loc[i, "Close"]
        time = df.loc[i, "Date"]

        # 开仓（从 0 → 1 或 0 → -1）
        if prev_sig == 0 and curr_sig != 0:
            position = curr_sig
            entry_price = price
            trades.append({
                "time": time,
                "action": "BUY" if curr_sig == 1 else "SELL",
                "price": price,
                "position": curr_sig,
                "pnl": 0
            })

        # 平仓（从 ±1 → 0）
        elif prev_sig != 0 and curr_sig == 0:
            pnl = (price - entry_price) * prev_sig * 1  # size=1
            trades.append({
                "time": time,
                "action": "CLOSE",
                "price": price,
                "position": 0,
                "pnl": pnl
            })
            position = 0
            entry_price = None

        # 反转（从 1 → -1 or -1 → 1）
        elif prev_sig != 0 and curr_sig != 0 and prev_sig != curr_sig:
            # 平先前仓
            pnl = (price - entry_price) * prev_sig
            trades.append({
                "time": time,
                "action": "REVERSE_CLOSE",
                "price": price,
                "position": 0,
                "pnl": pnl
            })

            # 开新反向仓
            position = curr_sig
            entry_price = price
            trades.append({
                "time": time,
                "action": "REVERSE_OPEN",
                "price": price,
                "position": curr_sig,
                "pnl": 0
            })

    return trades
