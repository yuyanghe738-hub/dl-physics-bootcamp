# Literature Map: 深度学习在物理中的应用

**Generated with ARIS-style `research-lit` workflow**  
**Date**: 2026-05-09

## Executive Summary

深度学习在物理中的应用可以分为四条主线：

1. **数据驱动物理建模**：从仿真或实验数据中学习规律，例如轨迹预测、材料性质预测、粒子物理事件分类。
2. **Physics-Informed Machine Learning**：把物理方程、守恒律、边界条件写进模型或 loss。
3. **神经算子 Neural Operators**：学习从函数到函数的映射，适合 PDE 多初值、多边界条件场景。
4. **参数反演与科学发现**：从观测数据推断未知参数、未知方程结构或异常现象。

对当前阶段，建议优先读 PINN 和 Neural Operator 的入门材料，不建议一开始进入高能物理或复杂材料数据库。

## Core Papers

### 1. Machine learning and the physical sciences

- **Authors**: Carleo et al.
- **Venue**: Reviews of Modern Physics, 2019
- **Link**: https://arxiv.org/abs/1903.10563
- **Why it matters**: 总览机器学习在物理科学中的应用，是该方向的经典综述。
- **Read first**:
  - Introduction
  - Overview of supervised/unsupervised/reinforcement learning
  - Sections related to quantum many-body, statistical physics, and materials
- **For you**: 用来建立地图，不需要逐页精读。

### 2. Physics-informed machine learning

- **Authors**: Karniadakis et al.
- **Venue**: Nature Reviews Physics, 2021
- **Link**: https://www.nature.com/articles/s42254-021-00314-5
- **Why it matters**: 解释如何把物理知识融入机器学习，是后续 PINN、PDE、反问题的核心入口。
- **Core idea**: 数据不足时，物理方程可以作为约束，减少模型胡乱拟合。
- **For you**: 这是最接近你“用深度学习做物理研究”目标的综述。

### 3. Scientific Machine Learning through Physics-Informed Neural Networks

- **Authors**: Cuomo et al.
- **Venue**: Journal of Scientific Computing, 2022
- **Link**: https://arxiv.org/abs/2201.05624
- **Why it matters**: 专门讲 PINN 的现状、方法、局限和未来问题。
- **Core idea**: 神经网络不只拟合数据，还要让预测函数满足微分方程残差。
- **For you**: 第一个可复现实验应该从这条线开始。

### 4. Learning nonlinear operators via DeepONet

- **Authors**: Lu et al.
- **Venue**: Nature Machine Intelligence, 2021
- **Link**: https://www.nature.com/articles/s42256-021-00302-5
- **Why it matters**: Neural Operator 的代表方向之一，目标是学习算子而不是单个函数。
- **Core idea**: PINN 往往针对一个具体 PDE 解；Neural Operator 希望学会“一类 PDE 解映射”。
- **For you**: 这是第二阶段方向，不应作为第一个项目。

### 5. Fourier Neural Operator for Parametric Partial Differential Equations

- **Authors**: Li et al.
- **Link**: https://arxiv.org/abs/2010.08895
- **Why it matters**: FNO 是神经算子的关键方法，常用于 PDE surrogate modeling。
- **For you**: 适合完成 PINN 小项目后再读。

## Recommended Reading Order

1. 先读 Carleo 2019 的引言和目录，建立方向地图。
2. 再读 Karniadakis 2021，理解“物理约束为什么重要”。
3. 然后读 Cuomo 2022，进入 PINN 具体训练结构。
4. 最后读 DeepONet/FNO，理解从“解一个方程”到“学习一类方程解算子”的升级。

## What To Ignore For Now

- 大规模高能物理事件重建
- 复杂材料数据库建模
- 图神经网络材料发现
- 扩散模型生成物理场
- 高维湍流 surrogate

这些方向有价值，但不适合作为你的第一个物理深度学习项目。

