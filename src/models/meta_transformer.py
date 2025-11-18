import torch
import torch.nn as nn
import numpy as np


class MetaTransformer(nn.Module):
    """
    Transformer-based meta model:
    输入：因子序列 (batch, seq_len, num_features)
    输出：策略权重 (batch, num_strategies)
    """
    def __init__(
        self,
        num_features: int,
        num_strategies: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        nhead: int = 4,
        seq_len: int = 30,
    ):
        super().__init__()

        self.seq_len = seq_len

        self.input_proj = nn.Linear(num_features, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(hidden_dim, num_strategies)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: (batch, seq_len, num_features)
        """
        h = self.input_proj(x)
        z = self.encoder(h)
        pooled = z[:, -1, :]  # 只取最后一步的 hidden state
        logits = self.fc(pooled)
        weights = self.softmax(logits)
        return weights
