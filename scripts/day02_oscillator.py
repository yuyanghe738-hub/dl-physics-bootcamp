from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

m = 1.0
k = 4.0
x0 = 1.0
v0 = 0.0
dt = 0.01
steps = 2000

t = np.arange(steps) * dt
x = np.zeros(steps)
v = np.zeros(steps)
x[0] = x0
v[0] = v0

for i in range(steps - 1):
    a = -(k / m) * x[i]
    v[i + 1] = v[i] + a * dt
    x[i + 1] = x[i] + v[i + 1] * dt

plt.figure(figsize=(8, 4))
plt.plot(t, x, label="position x(t)")
plt.xlabel("time (s)")
plt.ylabel("position")
plt.title("Day 2: Harmonic Oscillator")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "day02_oscillator.png", dpi=160)
plt.show()

