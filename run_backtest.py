import os

from src.data_loader import load_data
from src.strategy_ma import ma_crossover
from src.backtester.engine import backtest
from src.backtester.trade_log import generate_trade_log
from src.backtester.metrics import sharpe_ratio, max_drawdown, volatility
from src.optimizer.grid_search import grid_search_ma
from src.plot import plot_equity_and_drawdown, plot_entry_exit


DATA_PATH = "data/raw/data.csv"
CHART_DIR = "results/charts"


def run_single_backtest(df, short=10, long=50,
                        initial_capital=10000,
                        commission=0.0005, slippage=0.0002,
                        equity_path=None, entry_path=None,
                        label="Initial"):
    """
    跑一次简单的 MA 回测 + 打印指标 + 画图（可选）
    """
    print(f"\n=== {label} Strategy: MA({short}, {long}) ===")

    # 生成信号
    df = ma_crossover(df.copy(), short=short, long=long)

    # 回测
    df = backtest(df,
                  initial_capital=initial_capital,
                  commission=commission,
                  slippage=slippage)

    # 风险指标
    print("Risk Metrics:")
    print(f"  Sharpe Ratio       : {sharpe_ratio(df):.4f}")
    print(f"  Max Drawdown (MDD) : {max_drawdown(df):.4f}")
    print(f"  Volatility         : {volatility(df):.4f}")

    # 交易日志
    trades = generate_trade_log(df)
    print(f"  Total trades       : {len(trades)}")

    # 画图（如果给了路径）
    if equity_path is not None:
        plot_equity_and_drawdown(df, save_path=equity_path)
    if entry_path is not None:
        plot_entry_exit(df, trades, save_path=entry_path)

    return df, trades


def main():
    # 确保输出目录存在
    os.makedirs(CHART_DIR, exist_ok=True)

    # ========= 1. 加载数据 =========
    print("Loading data...")
    df_raw = load_data(DATA_PATH)

    # ========= 2. 初始策略回测 =========
    df_init, trades_init = run_single_backtest(
        df_raw,
        short=10,
        long=50,
        initial_capital=10000,
        commission=0.0005,
        slippage=0.0002,
        equity_path=os.path.join(CHART_DIR, "equity_drawdown_initial.png"),
        entry_path=os.path.join(CHART_DIR, "entry_exit_initial.png"),
        label="Initial",
    )

    # ========= 3. 参数优化（Grid Search） =========
    print("\nRunning parameter optimization (Grid Search)...")
    best, res_df = grid_search_ma(
        df_raw,  # 用原始数据做参数搜索更干净
        short_range=[5, 10, 20, 30],
        long_range=[50, 100, 150],
        commission=0.0005,
        slippage=0.0002,
        save_path=os.path.join(CHART_DIR, "heatmap_sharpe.png"),
    )

    print("\n=== Grid Search Results ===")
    print(res_df)

    print("\n=== Best Parameters ===")
    print(f"  Short MA : {best['short']}")
    print(f"  Long  MA : {best['long']}")
    print(f"  Sharpe   : {best['sharpe']:.4f}")

    # ========= 4. 使用最优参数再回测一遍 =========
    df_best, trades_best = run_single_backtest(
        df_raw,
        short=int(best["short"]),
        long=int(best["long"]),
        initial_capital=10000,
        commission=0.0005,
        slippage=0.0002,
        equity_path=os.path.join(CHART_DIR, "equity_drawdown_best.png"),
        entry_path=os.path.join(CHART_DIR, "entry_exit_best.png"),
        label="Optimized (Best Params)",
    )

    print("\n🎉 Clean run complete — Only best charts saved!")


if __name__ == "__main__":
    main()
