# run_multi_strategy.py

from src.config import (
    DATA_PATH,
    INITIAL_CAPITAL,
    COMMISSION,
    SLIPPAGE,
    RISK_FREQ,
    CHART_DIR,
    SYMBOL,
)

from src.data.loader import load_data
from src.strategies import STRATEGY_REGISTRY, STRATEGY_PARAM_MAP, apply_strategy
from src.backtester.engine import BacktestEngine
from src.backtester.trade_log import generate_trade_log
from src.backtester.metrics import sharpe_ratio, max_drawdown, volatility
from src.plot.equity import plot_equity_and_drawdown
from src.utils.helpers import print_section, ensure_dir

import pandas as pd
import os


def run_strategy(df_raw, name: str, params: dict):
    print_section(f"运行策略：{name}")
    print(f"参数：{params}")

    df_sig = apply_strategy(df_raw.copy(), name, **params)

    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        commission=COMMISSION,
        slippage=SLIPPAGE,
    )
    df_bt = engine.run(df_sig)
    trades = generate_trade_log(df_bt)

    sharpe = sharpe_ratio(df_bt, freq=RISK_FREQ)
    mdd = max_drawdown(df_bt)
    vol = volatility(df_bt, freq=RISK_FREQ)
    final_equity = df_bt["equity"].iloc[-1]

    # 图表保存
    eq_path = f"{CHART_DIR}/{name}_equity.png"
    ensure_dir(eq_path)
    plot_equity_and_drawdown(df_bt, save_path=eq_path)

    return {
        "strategy": name,
        "final_equity": final_equity,
        "sharpe": sharpe,
        "mdd": mdd,
        "vol": vol,
        "trades": len(trades),
        "chart": eq_path,
    }


def main():
    print_section("加载数据")
    df_raw = load_data(DATA_PATH)
    print(f"Rows: {len(df_raw)}, Columns: {list(df_raw.columns)}")

    results = []

    # 遍历你的策略列表
    for name, func in STRATEGY_REGISTRY.items():
        params = STRATEGY_PARAM_MAP[name]
        res = run_strategy(df_raw, name, params)
        results.append(res)

    print_section("多策略对比结果")

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by="sharpe", ascending=False)

    print(df_res)

    print("\n图表保存在 results/charts/ 下。")


if __name__ == "__main__":
    main()
