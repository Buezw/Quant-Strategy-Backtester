# src/meta/trainer.py
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F

from .transformer_weight import MetaTransformer


def train_meta_transformer(
    dataset,
    num_features,
    num_strats,
    lr=1e-3,
    batch_size=32,
    epochs=5,
    device="cpu",
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MetaTransformer(
        num_features=num_features,
        num_strats=num_strats
    ).to(device)

    opt = Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        total_loss = 0
        for X_seq, signals_now, y in loader:
            X_seq = X_seq.to(device)
            signals_now = signals_now.to(device)
            y = y.to(device)

            weights = model(X_seq)                        # (B, num_strats)
            ensemble_signal = (weights * signals_now).sum(dim=1)

            loss = F.mse_loss(ensemble_signal, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"[Epoch {ep+1}] loss = {total_loss:.6f}")

    return model
