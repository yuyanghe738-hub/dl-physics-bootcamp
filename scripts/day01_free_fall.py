from pathlib import Path

import matplotlib.pyplot as plt


out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

g = 9.8
y0 = 100.0
v0 = 0.0
dt = 0.05

times = []
positions = []
velocities = []

t = 0.0
y = y0
v = v0

while y >= 0:
    times.append(t)
    positions.append(y)
    velocities.append(v)

    v = v - g * dt
    y = y + v * dt
    t = t + dt

print(f"落地时间约为: {times[-1]:.2f} s")
print(f"落地前速度约为: {velocities[-1]:.2f} m/s")

plt.figure(figsize=(8, 4))
plt.plot(times, positions, label="height y(t)")
plt.xlabel("time (s)")
plt.ylabel("height (m)")
plt.title("Day 1: Free Fall")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "day01_free_fall.png", dpi=160)
plt.show()

