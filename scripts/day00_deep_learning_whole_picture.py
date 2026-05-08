from pathlib import Path

import matplotlib.pyplot as plt
import torch


out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(7)

# 1. Data: pretend these are experimental measurements from an unknown law.
x = torch.linspace(-3, 3, 120).reshape(-1, 1).to(device)
y_true = 0.5 * x**2 - 1.0 * x + 0.3
y = y_true + 0.25 * torch.randn_like(y_true)

# 2. Model: a tiny neural network that maps x -> y.
model = torch.nn.Sequential(
    torch.nn.Linear(1, 16),
    torch.nn.Tanh(),
    torch.nn.Linear(16, 16),
    torch.nn.Tanh(),
    torch.nn.Linear(16, 1),
).to(device)

# 3. Loss: how wrong the prediction is.
loss_fn = torch.nn.MSELoss()

# 4. Optimizer: how the model changes its parameters.
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

loss_history = []

# 5. Training loop: predict -> measure error -> compute gradients -> update.
for epoch in range(800):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    if epoch % 100 == 0:
        print(f"epoch={epoch:03d}, loss={loss.item():.6f}")

# 6. Prediction: use the trained model.
with torch.no_grad():
    y_learned = model(x)

print("\n一个深度学习任务的完整结构：")
print("数据 data -> 模型 model -> 预测 prediction -> 损失 loss -> 反向传播 backward -> 优化器 optimizer -> 重复训练")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].scatter(x.cpu(), y.cpu(), s=18, label="noisy data")
axes[0].plot(x.cpu(), y_true.cpu(), label="true law")
axes[0].plot(x.cpu(), y_learned.cpu(), "--", label="neural network")
axes[0].set_title("Learn a Hidden Physical Law")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(loss_history)
axes[1].set_title("Training Loss")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("MSE loss")
axes[1].grid(True)

plt.tight_layout()
plt.savefig(out_dir / "day00_deep_learning_whole_picture.png", dpi=160)
plt.show()

