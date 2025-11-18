# run_backtest.py

from src.config import (
    DATA_PATH,
    INITIAL_CAPITAL,
    COMMISSION,
    SLIPPAGE,
    RISK_FREQ,
    CHART_DIR,
    SYMBOL,
    STRATEGY_NAME,
    STRATEGY_PARAMS,
)

from src.data.loader import load_data
from src.strategies import apply_strategy
from src.backtester.engine import BacktestEngine
from src.backtester.trade_log import generate_trade_log
from src.backtester.metrics import sharpe_ratio, max_drawdown, volatility
from src.optimizer.grid_search import grid_search_ma
from src.plot.equity import plot_equity_and_drawdown
from src.plot.entry_exit import plot_entry_exit
from src.utils.helpers import print_section, time_block, ensure_dir


# ===============================
# 单次回测（不做Grid Search）
# ===============================
def run_single_backtest(df_raw, label: str):
    """
    使用 config 中指定的策略（STRATEGY_NAME）跑一次完整回测
    不画图，只打印指标
    """

    print_section(f"{label} 策略回测  ({STRATEGY_NAME})")
    print(f"使用参数：{STRATEGY_PARAMS}")

    # 1) 生成信号
    df_sig = apply_strategy(df_raw.copy(), STRATEGY_NAME, **STRATEGY_PARAMS)

    # 2) 回测
    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        commission=COMMISSION,
        slippage=SLIPPAGE,
    )
    df_bt = engine.run(df_sig)

    # 3) 交易日志
    trades = generate_trade_log(df_bt)

    # 4) 风险指标
    print("Risk Metrics:")
    print(f"  Sharpe Ratio : {sharpe_ratio(df_bt, freq=RISK_FREQ):.4f}")
    print(f"  Max Drawdown : {max_drawdown(df_bt):.4f}")
    print(f"  Volatility   : {volatility(df_bt, freq=RISK_FREQ):.4f}")
    print(f"  Total Trades : {len(trades)}")

    print("\nSample Trades (first 5):")
    for t in trades[:5]:
        print(" ", t)

    return df_bt, trades


# ===============================
# 主入口
# ===============================
def main():
    # --------------------------
    # 1. 加载数据
    # --------------------------
    print_section("加载数据")
    df_raw = load_data(DATA_PATH)
    print(f"Loaded data from: {DATA_PATH}")
    print(f"Rows: {len(df_raw)}, Columns: {list(df_raw.columns)}")

    # --------------------------
    # 2. baseline回测（按当前策略）
    # --------------------------
    df_init, trades_init = run_single_backtest(
        df_raw,
        label=f"Baseline ({SYMBOL})",
    )

    # --------------------------
    # 3. 如果不是 MA，则跳过 Grid Search
    # --------------------------
    if STRATEGY_NAME != "ma":
        print_section("当前策略不是 MA，跳过 Grid Search 参数优化")

        # 画图
        equity_path = f"{CHART_DIR}/equity_drawdown_{STRATEGY_NAME}.png"
        entry_path = f"{CHART_DIR}/entry_exit_{STRATEGY_NAME}.png"

        ensure_dir(equity_path)
        ensure_dir(entry_path)

        plot_equity_and_drawdown(df_init, save_path=equity_path)
        plot_entry_exit(df_init, save_path=entry_path)

        print("图表已输出：")
        print(f"  - {equity_path}")
        print(f"  - {entry_path}")
        return

    # --------------------------
    # 4. MA Grid Search
    # --------------------------
    print_section("Grid Search 参数优化 (MA)")

    with time_block("Grid Search (Sharpe)"):
        best, res_df = grid_search_ma(
            df_raw,
            short_range=MA_SHORT_RANGE,
            long_range=MA_LONG_RANGE,
            commission=COMMISSION,
            slippage=SLIPPAGE,
            save_path=f"{CHART_DIR}/heatmap_sharpe.png",
        )

    print("Grid Search 结果：")
    print(res_df)

    print("\nBest Parameters:")
    print(f"  short  = {best['short']}")
    print(f"  long   = {best['long']}")
    print(f"  sharpe = {best['sharpe']:.4f}")

    # --------------------------
    # 5. 用最优参数重新回测 + 图
    # --------------------------
    print_section("最优参数回测 + 图表输出")

    # 用 MA 的最佳参数重新生成信号
    df_best_sig = apply_strategy(
        df_raw.copy(),
        "ma",
        short_window=int(best["short"]),
        long_window=int(best["long"])
    )

    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        commission=COMMISSION,
        slippage=SLIPPAGE,
    )
    df_best = engine.run(df_best_sig)

    trades_best = generate_trade_log(df_best)

    equity_path = f"{CHART_DIR}/equity_drawdown_best.png"
    entry_path = f"{CHART_DIR}/entry_exit_best.png"

    ensure_dir(equity_path)
    ensure_dir(entry_path)

    plot_equity_and_drawdown(df_best, save_path=equity_path)
    plot_entry_exit(df_best, save_path=entry_path)

    print_section("完成")
    print("最佳参数策略已输出：")
    print(f"  - {equity_path}")
    print(f"  - {entry_path}")
    print(f"  - Sharpe (best) : {best['sharpe']:.4f}")


if __name__ == "__main__":
    main()
