from pathlib import Path

import matplotlib.pyplot as plt
import torch


out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(1)

t = torch.linspace(0, 10, 300).reshape(-1, 1).to(device)
x_true = torch.exp(-0.25 * t) * torch.cos(2.5 * t)

model = torch.nn.Sequential(
    torch.nn.Linear(1, 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 1),
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = torch.nn.MSELoss()

for epoch in range(1500):
    x_pred = model(t)
    loss = loss_fn(x_pred, x_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 300 == 0:
        print(f"epoch={epoch:4d} loss={loss.item():.6f}")

with torch.no_grad():
    x_pred = model(t).cpu()

plt.figure(figsize=(8, 4))
plt.plot(t.cpu(), x_true.cpu(), label="true damped oscillator")
plt.plot(t.cpu(), x_pred, "--", label="neural network")
plt.xlabel("time")
plt.ylabel("position")
plt.title("Day 6: Learn a Damped Oscillator")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "day06_damped_oscillator_learning.png", dpi=160)
plt.show()

