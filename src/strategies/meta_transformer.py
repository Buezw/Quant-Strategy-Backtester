import torch
import pandas as pd
import numpy as np

from src.meta.transformer_weight import MetaTransformer
from src.factors.factor_engine import generate_factors
from src.strategies.ma import ma_strategy
from src.strategies.rsi import rsi_strategy
from src.strategies.macd import macd_strategy
from src.strategies.bollinger import bollinger_strategy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def meta_transformer_strategy(df: pd.DataFrame,
                              model_path="models/meta_transformer.pt",
                              temperature=1.0,
                              **kwargs):

    ckpt = torch.load(model_path, map_location=DEVICE)

    state_dict = ckpt["state_dict"]
    factor_cols = ckpt["factor_cols"]
    strat_cols = ckpt["strat_cols"]
    seq_len = ckpt["seq_len"]

    input_dim = len(factor_cols)
    out_dim = len(strat_cols)

    df = generate_factors(df.copy())

    df["sig_ma"] = ma_strategy(df.copy())["signal"]
    df["sig_rsi"] = rsi_strategy(df.copy())["signal"]
    df["sig_macd"] = macd_strategy(df.copy())["signal"]
    df["sig_bollinger"] = bollinger_strategy(df.copy())["signal"]

    missing = [c for c in factor_cols + strat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少特征列: {missing}")

    X = torch.tensor(df[factor_cols].values, dtype=torch.float32)

    if len(X) < seq_len:
        df["signal"] = 0
        return df

    model = MetaTransformer(
        input_dim=input_dim,
        hidden_dim=64,
        n_heads=4,
        n_layers=2,
        out_dim=out_dim
    ).to(DEVICE)

    model.load_state_dict(state_dict)
    model.eval()

    preds = []

    for i in range(seq_len, len(X)):
        seq = X[i - seq_len:i].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            w = model(seq)
        w = torch.softmax(w / temperature, dim=-1).cpu().numpy()[0]
        preds.append(w)

    preds = np.array(preds)

    S = df[strat_cols].iloc[seq_len:].values

    out_sig = (preds * S).sum(axis=1)

    signals = np.zeros(len(df))
    signals[seq_len:] = out_sig
    df["signal"] = signals

    return df
