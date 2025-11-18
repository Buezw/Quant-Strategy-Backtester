import torch
import torch.nn as nn

class MetaTransformer(nn.Module):
    def __init__(
        self,
        input_dim=15,
        hidden_dim=64,
        n_heads=4,
        n_layers=2,
        out_dim=4,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.transformer(h)
        h = h.mean(dim=1)
        out = self.fc(h)
        return out
