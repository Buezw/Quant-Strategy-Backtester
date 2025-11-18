# src/meta/transformer_weight.py

import torch
import torch.nn as nn

class MetaTransformer(nn.Module):
    """
    与 checkpoint 完全一致的结构:
    input_proj → transformer → fc(out_dim=4)
    """
    def __init__(
        self,
        input_dim=15,     # MUST MATCH CHECKPOINT
        hidden_dim=64,
        n_heads=4,
        n_layers=2,
        out_dim=4,        # MUST MATCH CHECKPOINT
    ):
        super().__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim

        # 输入线性映射
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # 输出层（对应 4 个策略的权重）
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        h = self.input_proj(x)
        h = self.transformer(h)
        h = h.mean(dim=1)   # (batch, hidden)
        out = self.fc(h)    # (batch, out_dim)
        return out
