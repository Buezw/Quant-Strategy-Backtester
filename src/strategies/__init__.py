# src/strategies/__init__.py

from .ma import ma_strategy
from .rsi import rsi_strategy
from .macd import macd_strategy
from .bollinger import bollinger_strategy
from .breakout import breakout_strategy
from .momentum import momentum_strategy
from .zscore import zscore_strategy
from .meta_regime import meta_regime_strategy

# 新增
from .meta_transformer import meta_transformer_strategy

from .params import STRATEGY_PARAM_MAP


STRATEGY_REGISTRY = {
    "ma": ma_strategy,
    "rsi": rsi_strategy,
    "macd": macd_strategy,
    "bollinger": bollinger_strategy,
    "breakout": breakout_strategy,
    "momentum": momentum_strategy,
    "zscore": zscore_strategy,
    "meta_regime": meta_regime_strategy,

    # 新的 transformer 策略
    "meta_transformer": meta_transformer_strategy,
}


def apply_strategy(df, name: str, **params):
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"未知策略 '{name}', 可选: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name](df, **params)


__all__ = [
    "apply_strategy",
    "STRATEGY_REGISTRY",
    "STRATEGY_PARAM_MAP",
]
