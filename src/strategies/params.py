STRATEGY_PARAM_MAP = {
    "ma": {
        "short_window": 10,
        "long_window": 50,
    },
    "rsi": {
        "window": 14,
        "rsi_low": 30,
        "rsi_high": 70,
    },
    "macd": {
        "fast": 12,
        "slow": 26,
        "signal_window": 9,
    },
    "bollinger": {
        "window": 20,
        "num_std": 2,
    },
    "breakout": {
        "high_window": 20,
        "low_window": 10,
    },
    "momentum": {
        "lookback": 20,
    },
    "zscore": {
        "window": 20,
        "z_entry": 2.0,
    },
    "meta_regime": {
        "trend_ma_short": 50,
        "trend_ma_long": 200,
        "trend_threshold": 0.01,   

        "trend_mode": "momentum",    # "ma" / "breakout" / "momentum"
        "ma_short_window": 20,
        "ma_long_window": 100,
        "breakout_high_window": 20,
        "breakout_low_window": 10,
        "momentum_lookback": 20,

        "zscore_window": 20,
        "zscore_entry": 2.0,
    },
    "meta_xgb_weight": {
        "horizon": 1,
        "retrain": True,
        "temperature": 1.0,
    }
}



