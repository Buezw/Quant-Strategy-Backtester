from .ma import ma_strategy
from .rsi import rsi_strategy
from .macd import macd_strategy
from .bollinger import bollinger_strategy
from .breakout import breakout_strategy
from .momentum import momentum_strategy
from .zscore import zscore_strategy
from .meta_regime import meta_regime_strategy

# ⚠️ 必须新增的 import
from .meta_xgb_weight import meta_xgb_weight_strategy

# 参数表
from .params import STRATEGY_PARAM_MAP


# ===========================
# 策略注册表
# ===========================
STRATEGY_REGISTRY = {
    "ma": ma_strategy,
    "rsi": rsi_strategy,
    "macd": macd_strategy,
    "bollinger": bollinger_strategy,
    "breakout": breakout_strategy,
    "momentum": momentum_strategy,
    "zscore": zscore_strategy,
    "meta_regime": meta_regime_strategy,

    # ⚠️ 必须新增这一行
    "meta_xgb_weight": meta_xgb_weight_strategy,
}


# ===========================
# 统一策略入口
# ===========================
def apply_strategy(df, name: str, **params):
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"未知策略 '{name}', 可选: {list(STRATEGY_REGISTRY.keys())}")

    return STRATEGY_REGISTRY[name](df, **params)


__all__ = [
    "STRATEGY_REGISTRY",
    "STRATEGY_PARAM_MAP",
    "apply_strategy",
]
