

DATA_PATH = "data/raw/data.csv"

RESULT_DIR = "results"
CHART_DIR = f"{RESULT_DIR}/charts"



INITIAL_CAPITAL = 10_000.0
COMMISSION = 0.0005
SLIPPAGE = 0.0002
RISK_FREQ = "1h"

SYMBOL = "NVDA"


from src.strategies import STRATEGY_PARAM_MAP
STRATEGY_NAME = "meta_transformer"
STRATEGY_PARAMS = {
    "model_path": "models/meta_transformer.pt",
    "temperature": 1.0,
}

MA_SHORT_RANGE = [5, 10, 20, 30]
MA_LONG_RANGE  = [50, 100, 150]
