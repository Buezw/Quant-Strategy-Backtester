# ==========================
# 数据 & 路径
# ==========================

DATA_PATH = "data/raw/data.csv"

RESULT_DIR = "results"
CHART_DIR = f"{RESULT_DIR}/charts"

# ==========================
# 回测参数
# ==========================

INITIAL_CAPITAL = 10_000.0
COMMISSION = 0.0005
SLIPPAGE = 0.0002
RISK_FREQ = "1h"

SYMBOL = "NVDA"

# ==========================
# 策略选择
# ==========================

from src.strategies import STRATEGY_PARAM_MAP

STRATEGY_NAME = "meta_xgb_weight"
STRATEGY_PARAMS = STRATEGY_PARAM_MAP[STRATEGY_NAME]


# ==========================
# MA Grid Search（仅 MA 用）
# ==========================

MA_SHORT_RANGE = [5, 10, 20, 30]
MA_LONG_RANGE  = [50, 100, 150]
