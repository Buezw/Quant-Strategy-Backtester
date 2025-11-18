def cost_analysis(df):
    return {
        "total_cost": df["cost"].sum(),
        "commission": df["commission_cost"].sum(),
        "slippage": df["slippage_cost"].sum(),
        "trades": int(df["trade_flag"].sum()),
    }
