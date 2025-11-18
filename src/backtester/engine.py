import pandas as pd
import numpy as np


class BacktestEngine:

    def __init__(
        self,
        initial_capital: float = 10_000.0, #Initial_capital
        commission: float = 0.0005,        #Commission
        slippage: float = 0.0002,          #Slippage
    ):
        self.initial_capital = float(initial_capital)
        self.commission = float(commission)
        self.slippage = float(slippage)

    def _validate_input(self, df: pd.DataFrame) -> None:
        required = ["Close", "signal"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"BacktestEngine.run() Missing: {missing}")

    def _prepare_signal_and_position(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = df["signal"].fillna(0).astype(float)

        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)
        else:
            df = df.sort_index().reset_index(drop=True)

        df["position"] = df["signal"].shift(1).fillna(0.0)

        return df

    def _compute_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df["price_ret"] = df["Close"].pct_change().fillna(0.0)
        df["strategy_ret"] = df["position"] * df["price_ret"]
        return df

    def _compute_costs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算交易成本：
        - trade_flag: 仓位是否发生变化（换仓）
        - commission_cost: 手续费
        - slippage_cost: 滑点
        - cost: 总成本（commission + slippage）
        这里用“单位名义资金比例成本”，直接从收益里扣。
        """
        df["trade_flag"] = df["signal"].diff().abs().fillna(0.0)


        df["commission_cost"] = df["trade_flag"] * self.commission
        df["slippage_cost"] = df["trade_flag"] * self.slippage
        df["cost"] = df["commission_cost"] + df["slippage_cost"]

        return df

    def _compute_equity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算净收益 & 净值：
        - net_ret: 策略收益 - 成本
        - equity: 资金曲线
        """
        df["net_ret"] = df["strategy_ret"] - df["cost"]
        df["equity"] = self.initial_capital * (1.0 + df["net_ret"]).cumprod()
        return df

    # ==============================
    # 对外主接口
    # ==============================

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        运行完整回测流程。
        :param df: 至少包含 ['Close', 'signal'] 的 DataFrame
        :return: 附加回测结果列后的 DataFrame（原 df 的 copy，不会改原数据）
        """
        self._validate_input(df)

        df = df.copy()
        df = self._prepare_signal_and_position(df)
        df = self._compute_returns(df)
        df = self._compute_costs(df)
        df = self._compute_equity(df)

        return df


# =========================================
# 方便调用的函数式封装（保持你原来的习惯）
# =========================================

def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 10_000.0,
    commission: float = 0.0005,
    slippage: float = 0.0002,
) -> pd.DataFrame:
    """
    快捷函数：不想管类的时候直接用这个。
    """
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage,
    )
    return engine.run(df)
