# src/strategies/meta_xgb_weight.py

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# 这里假定你有一个因子引擎，如果还没有，可以先简单写个占位版
from src.factors.factor_engine import generate_factors

from .ma import ma_strategy
from .rsi import rsi_strategy
from .macd import macd_strategy
from .bollinger import bollinger_strategy


def _compute_forward_return(close: pd.Series, signal: pd.Series, horizon: int = 1):
    """
    给定收盘价和策略持仓 signal，近似算下一步策略收益：
      ret_{t+1} ≈ signal_t * (Close_{t+1}/Close_t - 1)
    """
    px_ret_fwd = close.pct_change(horizon).shift(-horizon)
    strat_ret_fwd = signal * px_ret_fwd
    return strat_ret_fwd


def _build_meta_dataset(
    df_raw: pd.DataFrame,
    horizon: int = 1,
):
    """
    构建 Meta-XGB 训练数据集：
      X: 因子特征
      Y: 每条策略的未来一步收益 [ret_ma, ret_rsi, ret_macd, ret_boll]

    返回：
      df_fac       : 因子表（已对齐）
      strat_names  : 底层策略名字列表
      ret_matrix   : shape (N, K) 的未来收益矩阵
      feature_cols : 因子列名
    """

    # 1) 因子
    df_fac = generate_factors(df_raw.copy())
    df_fac = df_fac.dropna()

    # 2) 跑底层策略
    df_ma = ma_strategy(df_raw.copy())
    df_rsi = rsi_strategy(df_raw.copy())
    df_macd = macd_strategy(df_raw.copy())
    df_boll = bollinger_strategy(df_raw.copy())

    # 3) 对齐索引
    idx = (
        df_fac.index
        .intersection(df_ma.index)
        .intersection(df_rsi.index)
        .intersection(df_macd.index)
        .intersection(df_boll.index)
    )

    df_fac = df_fac.loc[idx]
    df_ma = df_ma.loc[idx]
    df_rsi = df_rsi.loc[idx]
    df_macd = df_macd.loc[idx]
    df_boll = df_boll.loc[idx]

    close = df_raw.loc[idx, "Close"]

    # 4) 计算每条策略的未来收益
    ret_ma = _compute_forward_return(close, df_ma["signal"], horizon=horizon)
    ret_rsi = _compute_forward_return(close, df_rsi["signal"], horizon=horizon)
    ret_macd = _compute_forward_return(close, df_macd["signal"], horizon=horizon)
    ret_boll = _compute_forward_return(close, df_boll["signal"], horizon=horizon)

    df_meta = df_fac.copy()
    df_meta["ret_ma"] = ret_ma
    df_meta["ret_rsi"] = ret_rsi
    df_meta["ret_macd"] = ret_macd
    df_meta["ret_boll"] = ret_boll

    # 去掉有 NaN 的行
    df_meta = df_meta.dropna(subset=["ret_ma", "ret_rsi", "ret_macd", "ret_boll"])

    strat_names = ["ma", "rsi", "macd", "bollinger"]
    ret_matrix = df_meta[["ret_ma", "ret_rsi", "ret_macd", "ret_boll"]].values

    # 特征列：去掉收益列
    drop_cols = ["ret_ma", "ret_rsi", "ret_macd", "ret_boll"]
    feature_cols = [c for c in df_meta.columns if c not in drop_cols]

    X = df_meta[feature_cols].values

    return (
        df_meta.index,   # 对应的时间索引
        X,
        ret_matrix,
        feature_cols,
        strat_names,
    )


def _train_meta_xgb_weight_models(
    df_raw: pd.DataFrame,
    horizon: int = 1,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    训练一组 XGBRegressor，每条策略一个回归模型，预测其下一步收益。
    
    返回：
      models       : {strategy_name: xgb_model}
      scaler       : 标准化器
      feature_cols : 特征列名
      idx          : 训练数据对应的时间索引
    """

    idx, X, Y, feature_cols, strat_names = _build_meta_dataset(
        df_raw,
        horizon=horizon,
    )

    # 时间顺序切分 train / test
    n = len(X)
    split = int(n * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}

    from sklearn.metrics import mean_squared_error

    for i, strat in enumerate(strat_names):
        y_train = Y_train[:, i]
        y_test = Y_test[:, i]

        model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=random_state,
        )

        model.fit(X_train_scaled, y_train)

        y_pred_test = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred_test)
        print(f"[Meta-XGB] Strategy {strat} Test MSE: {mse:.6e}")

        models[strat] = model

    return models, scaler, feature_cols, idx


def meta_xgb_weight_strategy(
    df: pd.DataFrame,
    horizon: int = 1,
    retrain: bool = True,
    models: dict | None = None,
    scaler: StandardScaler | None = None,
    feature_cols: list[str] | None = None,
    temperature: float = 1.0,
) -> pd.DataFrame:
    """
    XGBoost 权重版 Meta 策略：
      1. 对每条底层策略（MA/RSI/MACD/BOLL）训练一个 XGB 回归器，
         预测未来一步策略收益。
      2. 将预测收益通过 softmax 转成权重。
      3. 用权重加权各策略当前 signal，得到 final_signal。

    返回的 df_out 至少包含：
      - "signal"         : 最终合成信号
      - "signal_ma"...   : 各底层策略信号
      - "weight_ma"...   : 各底层策略权重
    """

    df_raw = df.copy()

    # 1) 训练模型（或使用外部传入的已训练模型）
    if retrain or models is None or scaler is None or feature_cols is None:
        models, scaler, feature_cols, idx = _train_meta_xgb_weight_models(
            df_raw,
            horizon=horizon,
        )
    else:
        # 如果使用外部模型，需要重新构造特征，但保证列名一致
        from src.factors.factor_engine import generate_factors

        df_fac = generate_factors(df_raw.copy()).dropna()
        idx = df_fac.index
        X = df_fac.loc[idx, feature_cols].values
        # NOTE: 下面会重用这个 X_scaled
        X_scaled = scaler.transform(X)

    strat_names = list(models.keys())

    # 2) 再算一遍因子表（与训练一致），得到 X_scaled
    from src.factors.factor_engine import generate_factors

    df_fac = generate_factors(df_raw.copy()).dropna()
    df_fac = df_fac.loc[idx]  # 对齐索引
    X = df_fac[feature_cols].values
    X_scaled = scaler.transform(X)

    # 3) 底层策略信号（在相同 idx 上）
    df_ma = ma_strategy(df_raw.copy()).loc[idx]
    df_rsi = rsi_strategy(df_raw.copy()).loc[idx]
    df_macd = macd_strategy(df_raw.copy()).loc[idx]
    df_boll = bollinger_strategy(df_raw.copy()).loc[idx]

    strat_signal_map = {
        "ma": df_ma["signal"],
        "rsi": df_rsi["signal"],
        "macd": df_macd["signal"],
        "bollinger": df_boll["signal"],
    }

    # 4) 用每个模型预测未来收益 → 预测矩阵 pred_ret: (N, K)
    N = X_scaled.shape[0]
    K = len(strat_names)
    pred_ret = np.zeros((N, K), dtype=float)

    for j, strat in enumerate(strat_names):
        model = models[strat]
        pred_ret[:, j] = model.predict(X_scaled)

    # 5) 用 temperature 控制 softmax 平滑度，得到权重矩阵 (N, K)
    if temperature <= 0:
        temperature = 1.0

    scores = pred_ret / temperature
    # 数值稳定 softmax
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    # 6) 组合最终 signal
    df_out = df_raw.loc[idx].copy()

    # 保存各策略 signal & weight
    for j, strat in enumerate(strat_names):
        sig = strat_signal_map[strat]
        df_out[f"signal_{strat}"] = sig.values
        df_out[f"weight_{strat}"] = weights[:, j]

    # final_signal = Σ w_i * signal_i
    final_signal = np.zeros(N, dtype=float)
    for j, strat in enumerate(strat_names):
        final_signal += weights[:, j] * df_out[f"signal_{strat}"].values

    # 你可以直接用连续仓位，或者再 threshold 成 { -1,0,1 }
    df_out["signal_raw"] = final_signal

    # 简单 threshold 版（可按需要修改）
    df_out["signal"] = 0.0
    df_out.loc[final_signal > 0.1, "signal"] = 1.0
    df_out.loc[final_signal < -0.1, "signal"] = -1.0

    return df_out
