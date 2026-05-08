from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)
rng = np.random.default_rng(42)

true_a = 2.5
true_b = 1.2
x = np.linspace(0, 10, 40)
noise = rng.normal(0, 1.5, size=x.shape)
y = true_a * x + true_b + noise

A = np.vstack([x, np.ones_like(x)]).T
estimated_a, estimated_b = np.linalg.lstsq(A, y, rcond=None)[0]

print(f"真实参数: a={true_a:.3f}, b={true_b:.3f}")
print(f"估计参数: a={estimated_a:.3f}, b={estimated_b:.3f}")

plt.figure(figsize=(8, 4))
plt.scatter(x, y, label="noisy data")
plt.plot(x, true_a * x + true_b, label="true law")
plt.plot(x, estimated_a * x + estimated_b, "--", label="estimated law")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Day 3: Recover a Physical Parameter")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "day03_linear_regression_numpy.png", dpi=160)
plt.show()

