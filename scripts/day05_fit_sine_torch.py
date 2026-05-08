from pathlib import Path

import matplotlib.pyplot as plt
import torch


out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)

x = torch.linspace(-3.14, 3.14, 200).reshape(-1, 1).to(device)
y = torch.sin(x)

model = torch.nn.Sequential(
    torch.nn.Linear(1, 32),
    torch.nn.Tanh(),
    torch.nn.Linear(32, 32),
    torch.nn.Tanh(),
    torch.nn.Linear(32, 1),
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

for epoch in range(1000):
    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"epoch={epoch:4d} loss={loss.item():.6f}")

with torch.no_grad():
    pred = model(x).cpu()

plt.figure(figsize=(8, 4))
plt.plot(x.cpu(), y.cpu(), label="true sin(x)")
plt.plot(x.cpu(), pred, "--", label="neural network")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Day 5: Fit sin(x) with PyTorch")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "day05_fit_sine_torch.png", dpi=160)
plt.show()

