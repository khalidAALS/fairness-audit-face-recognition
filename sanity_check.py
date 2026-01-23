import torch
print("torch:", torch.__version__)
print("mps available:", torch.backends.mps.is_available())
print("mps built:", torch.backends.mps.is_built())

device = "mps" if torch.backends.mps.is_available() else "cpu"
x = torch.randn(1024, 1024, device=device)
y = x @ x.T
print("device:", device, "result mean:", y.mean().item())
