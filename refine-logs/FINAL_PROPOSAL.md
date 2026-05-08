# Final Proposal: PINN 求解一维热方程

**Generated with ARIS-style `research-refine` workflow**  
**Date**: 2026-05-09

## Problem Anchor

用户需要一个能快速上手、真正体现深度学习结构、同时直接连接物理研究的项目。第一个项目必须避免复杂数据集和大模型，优先选择可解析、可可视化、可调试的物理系统。

## Proposed Method

训练一个多层感知机 `u_theta(x, t)`，输入空间位置 `x` 和时间 `t`，输出温度 `u`。

热方程：

```text
u_t = alpha * u_xx
```

训练目标由三部分组成：

```text
total_loss = PDE residual loss + initial condition loss + boundary condition loss
```

其中：

- PDE residual loss：让模型输出满足热方程
- initial condition loss：让 `t=0` 时的预测满足初始温度分布
- boundary condition loss：让边界点满足给定边界条件

## Why This Is the Right First Project

1. 它不是普通函数拟合，而是完整 PINN。
2. 它能解释深度学习训练闭环。
3. 它有明确物理方程。
4. 它可视化容易，失败也容易诊断。
5. 它可以扩展成 MLP vs PINN、不同数据量、不同边界条件的研究型比较。

## Minimal Technical Design

### Model

```text
input:  (x, t)
MLP:    Linear -> Tanh -> Linear -> Tanh -> Linear
output: u(x, t)
```

### Loss Terms

```text
L_pde = mean((u_t - alpha * u_xx)^2)
L_ic  = mean((u(x,0) - sin(pi*x))^2)
L_bc  = mean(u(0,t)^2 + u(1,t)^2)
L     = L_pde + L_ic + L_bc
```

### Domain

```text
x in [0, 1]
t in [0, 1]
alpha = 0.1
initial condition: u(x,0) = sin(pi*x)
boundary condition: u(0,t)=0, u(1,t)=0
```

### Analytic Reference

For this setup:

```text
u(x,t) = exp(-alpha*pi^2*t) * sin(pi*x)
```

This gives a direct correctness check.

## Expected Deliverables

- `scripts/day08_pinn_heat_equation.py`
- loss curve
- predicted solution heatmap
- analytic solution heatmap
- absolute error heatmap
- short explanation document

## Main Risk

PINN training can be unstable if loss terms are imbalanced. Start with equal weights, then inspect individual loss components.

## Success Condition

The project is successful if:

- PDE residual decreases
- initial/boundary conditions are visually satisfied
- predicted heatmap resembles analytic solution
- mean absolute error is reported
- user can explain each loss term

