import torch

ckpt = torch.load("models/meta_transformer.pt", map_location="cpu")
sd = ckpt["state_dict"]

print("="*50)
print("input_proj.weight shape:", sd["input_proj.weight"].shape)
print("fc.weight shape:", sd["fc.weight"].shape)
print("-"*50)
print("factor_cols:", ckpt["factor_cols"])
print("strat_cols:", ckpt["strat_cols"])
print("seq_len:", ckpt["seq_len"])
print("="*50)
