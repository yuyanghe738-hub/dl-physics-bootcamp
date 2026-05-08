import torch


x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2 * x + 1

y.backward()

print("x =", x.item())
print("y = x^2 + 2x + 1 =", y.item())
print("dy/dx =", x.grad.item())
print("解析答案应该是 2x + 2 =", 2 * x.item() + 2)

