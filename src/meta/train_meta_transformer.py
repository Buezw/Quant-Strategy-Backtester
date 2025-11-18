import torch
import pandas as pd

from src.config import DATA_PATH
from src.data.loader import load_data
from src.factors.factor_engine import generate_factors

from src.strategies.ma import ma_strategy
from src.strategies.rsi import rsi_strategy
from src.strategies.macd import macd_strategy
from src.strategies.bollinger import bollinger_strategy

from src.meta.dataset import MetaSequenceDataset
from src.meta.trainer import train_meta_transformer


def main():
    print(">>> 加载原始数据")
    df_raw = load_data(DATA_PATH)

    print(">>> 生成因子")
    df_fac = generate_factors(df_raw.copy())

    print(">>> 计算基础策略信号 (MA / RSI / MACD / Bollinger)")

    strat_funcs = {
        "ma": ma_strategy,
        "rsi": rsi_strategy,
        "macd": macd_strategy,
        "bollinger": bollinger_strategy,
    }

    for name, func in strat_funcs.items():
        df_sig = func(df_fac.copy())
        df_fac[f"sig_{name}"] = df_sig["signal"]

    df_fac = df_fac.dropna().copy()

    strat_cols = [f"sig_{name}" for name in strat_funcs.keys()]

    exclude = set(["Open", "High", "Low", "Close", "Volume"] + strat_cols)
    factor_cols = [
        c for c in df_fac.columns
        if (c not in exclude) and (df_fac[c].dtype != "O")
    ]

    print("使用的因子列:", factor_cols)
    print("使用的策略列:", strat_cols)

    dataset = MetaSequenceDataset(
        df_fac,
        strat_cols=strat_cols,
        factor_cols=factor_cols,
        seq_len=32,
        horizon=1,
    )

    print(f"Dataset 长度: {len(dataset)} 样本")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> 使用设备: {device}")

    model = train_meta_transformer(
        dataset,
        num_features=len(factor_cols),
        num_strats=len(strat_cols),
        lr=1e-3,
        batch_size=32,
        epochs=8,
        device=device,
    )

    save_path = "models/meta_transformer.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "factor_cols": factor_cols,
        "strat_cols": strat_cols,
        "seq_len": 32,
        "horizon": 1,
    }, save_path)

    print(f"Meta-Transformer 模型已保存到: {save_path}")


if __name__ == "__main__":
    main()
