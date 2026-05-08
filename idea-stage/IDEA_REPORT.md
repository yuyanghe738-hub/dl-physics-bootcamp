# Idea Report: 适合入门的物理深度学习课题

**Generated with ARIS-style `idea-discovery` workflow**  
**Date**: 2026-05-09

## Ranking Criteria

- 能否快速上手写代码
- 是否展示真实深度学习结构
- 是否和物理问题直接相关
- 是否能逐步扩展成研究问题
- 是否适合本机 Mac 小规模运行

## Candidate 1: 用 PINN 求解一维热方程

**Problem**: 给定一维热方程和边界条件，让神经网络学习温度场 `u(x, t)`。

**Why this is best first**:

- PDE 简单
- 物理意义清楚
- 可以画出温度随时间扩散的图
- 能完整展示 PINN 结构：数据 loss、方程残差 loss、边界 loss、初值 loss

**Minimum project output**:

- 训练 loss 曲线
- 预测温度场图
- 与解析解或有限差分解的误差图

**Risk**:

- PINN 训练可能不稳定。解决方式是先从少量点和简单边界条件开始。

**Decision**: 作为第一个研究型项目。

## Candidate 2: 用神经网络学习阻尼振子并反推阻尼系数

**Problem**: 从带噪声轨迹数据中预测位置，并反推出阻尼参数。

**Why useful**:

- 和 Day 06 已有脚本自然衔接
- 比 PINN 更容易
- 适合理解“参数反演”

**Minimum project output**:

- 轨迹拟合图
- 参数估计结果
- 不同噪声水平下的误差表

**Decision**: 作为 PINN 前的过渡项目，建议 3-5 天完成。

## Candidate 3: 比较纯数据神经网络与 PINN

**Problem**: 在数据很少时，比较普通 MLP 和 PINN 谁更稳。

**Why research-like**:

- 有明确 claim：物理约束是否提升小数据泛化？
- 可以做 ablation：无 PDE loss、弱 PDE loss、强 PDE loss

**Minimum project output**:

- 不同训练点数量下的误差曲线
- loss 组成分析
- 可视化预测场

**Decision**: 作为 Candidate 1 的升级版。

## Candidate 4: 用 FNO 学习一类热方程解

**Problem**: 输入不同初始温度分布，输出未来温度场。

**Why interesting**:

- 连接 Neural Operator
- 比单个 PINN 更接近现代科学机器学习

**Risk**:

- 实现复杂度明显更高
- 需要生成较多 PDE 数据

**Decision**: 第二阶段项目，不做第一个。

## Recommended Path

```text
阻尼振子参数反演
-> PINN 求解一维热方程
-> 比较 MLP vs PINN 小数据泛化
-> FNO/DeepONet 学习一类 PDE 解
```

## Chosen First Research Project

**Title**: Physics-Informed Neural Network for the 1D Heat Equation

**One-sentence thesis**: 用一个小型神经网络表示 `u(x,t)`，并通过 PDE 残差、初值和边界条件共同训练，使模型在少量监督数据下学习符合物理规律的温度场。

