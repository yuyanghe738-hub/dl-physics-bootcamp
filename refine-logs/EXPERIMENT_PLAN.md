# Experiment Plan

**Problem**: 用深度学习学习物理规律，并进入 PINN 方向。  
**Method Thesis**: 用神经网络表示一维热方程解 `u(x,t)`，通过物理方程残差、初值和边界条件共同训练。  
**Date**: 2026-05-09

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| C1: PINN 可以学习热方程解 | 证明深度学习能和物理方程结合 | 预测热图接近解析解，误差可量化 | B1 |
| C2: 物理约束能提升小数据场景 | 连接科研问题，而非只做 demo | 少量监督点下，PINN 优于普通 MLP | B2 |

## Paper Storyline

- Main paper/report must prove: PINN 的训练结构和物理约束确实起作用。
- Appendix can support: 不同学习率、网络宽度、采样点数量对结果的影响。
- Experiments intentionally cut: 高维 PDE、复杂边界、FNO、大规模数据集。

## Experiment Blocks

### Block 1: Minimal PINN Heat Equation

- **Claim tested**: C1
- **Why this block exists**: 建立第一个完整物理深度学习闭环。
- **Dataset / split / task**: 在 `[0,1] x [0,1]` 上采样 collocation points、initial points、boundary points。
- **Compared systems**: PINN only。
- **Metrics**: mean absolute error, MSE, visual error heatmap。
- **Setup details**: MLP, Tanh, Adam, 2000-5000 epochs, alpha=0.1。
- **Success criterion**: MAE 明显小于初始随机模型；热图接近解析解。
- **Failure interpretation**: loss 权重、采样点、学习率或模型容量需要调整。
- **Table / figure target**: Figure 1: predicted vs analytic vs error。
- **Priority**: MUST-RUN

### Block 2: MLP vs PINN Under Sparse Data

- **Claim tested**: C2
- **Why this block exists**: 证明物理约束不是装饰。
- **Dataset / split / task**: 只给少量监督点，比较普通 MLP 和 PINN。
- **Compared systems**:
  - Plain MLP: only data loss
  - PINN: data + PDE + boundary/initial losses
- **Metrics**: test MAE across full grid。
- **Setup details**: same architecture, same optimizer, same epochs。
- **Success criterion**: 少量数据时 PINN 泛化更好。
- **Failure interpretation**: 如果 PINN 不优，检查 PDE residual 采样和 loss 权重。
- **Table / figure target**: Table 1: MAE vs number of supervised points。
- **Priority**: MUST-RUN after Block 1

### Block 3: Ablation on Loss Terms

- **Claim tested**: C1/C2
- **Why this block exists**: 理解每个 loss 项的作用。
- **Compared systems**:
  - without PDE loss
  - without boundary loss
  - without initial loss
  - full PINN
- **Metrics**: MAE, boundary violation, PDE residual。
- **Priority**: NICE-TO-HAVE

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | 环境确认 | run PyTorch + autograd test | 能计算 `u_t` 和 `u_xx` | 5 min | autograd 维度错误 |
| M1 | 最小 PINN 跑通 | Block 1 one seed | loss 下降且图像合理 | 30-60 min | loss 不稳定 |
| M2 | 误差评估 | analytic grid comparison | 输出 MAE 和 heatmap | 15 min | 画图或 mesh shape 错 |
| M3 | MLP vs PINN | Block 2 | PINN 小数据更稳 | 1-2 h | 对比不公平 |
| M4 | loss ablation | Block 3 | 明确每个 loss 贡献 | 1-2 h | 实验数量膨胀 |

## Compute and Data Budget

- **Total estimated GPU-hours**: 本机小模型，预计 1-4 小时内完成第一版。
- **Data preparation needs**: 无外部数据；全部由采样和解析解生成。
- **Human evaluation needs**: 观察图像是否符合物理直觉。
- **Biggest bottleneck**: 理解 autograd 如何计算 PDE 残差。

## First Three Runs

1. `R001`: 验证 `x,t -> u` 网络能 forward。
2. `R002`: 验证 PyTorch 能计算 `u_t` 和 `u_xx`。
3. `R003`: 训练完整 PINN 并生成三张图。

## Final Checklist

- [ ] Main PINN script exists
- [ ] Loss curve saved
- [ ] Predicted heatmap saved
- [ ] Analytic heatmap saved
- [ ] Error heatmap saved
- [ ] MAE printed
- [ ] User can explain PDE loss, IC loss, BC loss

