# Research Brief: 深度学习在物理中的应用

**Date**: 2026-05-09

## Problem Statement

目标不是泛泛学习深度学习，而是建立一条能逐步走向物理研究的路线：先理解神经网络训练闭环，再用可控物理问题建立实验能力，最后进入 Physics-Informed Neural Networks、Neural Operators、参数反演等科学机器学习方向。

当前最大问题是基础跨度过大：用户没有 Python 基础，也没有深度学习系统训练经验。如果直接读论文或教材，会缺少代码闭环；如果只做 Python 仿真，又会偏离“深度学习”的核心结构。因此路线必须从最小完整深度学习任务开始，并尽快连接到物理问题。

## Background

- **Field**: Scientific Machine Learning / Deep Learning for Physical Sciences
- **Sub-area**: PINN、神经算子、物理参数反演、物理数据拟合
- **Key entry papers**:
  - Carleo et al., 2019, *Machine learning and the physical sciences*
  - Karniadakis et al., 2021, *Physics-informed machine learning*
  - Cuomo et al., 2022, *Scientific Machine Learning through Physics-Informed Neural Networks*
  - Lu et al., 2021, *Learning nonlinear operators via DeepONet*

## Constraints

- **基础**: Python 零基础，深度学习概念很浅
- **学习方式**: 项目驱动，每天必须能跑代码、改参数、看到输出
- **算力**: 本机 Mac，PyTorch 可用，MPS 可用；适合小模型和教学级实验
- **短期目标**: 2 周内理解深度学习闭环，1 个月内完成第一个物理深度学习小项目
- **长期目标**: 将深度学习用于物理方向研究

## What I Am Looking For

- [x] 从零开始的学习路径
- [x] 可执行的小课题
- [x] 物理方向文献地图
- [x] 第一个可实现实验计划
- [ ] 直接冲顶会论文
- [ ] 大规模 GPU 实验

## Non-Goals

- 不追求一开始数学完全扎实
- 不做大型模型训练
- 不做与物理无关的纯 CV/NLP 入门路线
- 不把 PINN 当黑箱套模板，必须理解 loss、梯度、边界条件

