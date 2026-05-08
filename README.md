# Deep Learning for Physics Bootcamp

这是一个从零开始、每天都能敲代码的小训练营。主线目标：

1. 先会用 Python 表达物理问题。
2. 再用 NumPy/Matplotlib 做仿真和画图。
3. 然后进入 PyTorch，理解张量、自动求导、训练循环。
4. 最后开始做物理中的深度学习：拟合、参数反演、PINN。

## 环境

你的机器上已经有可用环境：

```bash
conda activate physics
```

检查环境：

```bash
python scripts/check_env.py
```

## 每天怎么学

每天只做三件事：

1. 跑当天脚本。
2. 改 1-3 个参数，再跑一次。
3. 写下你看到的变化。

第一天开始：

```bash
cd /Users/heyuyang/Documents/Codex/2026-05-08/python/dl-physics-bootcamp
conda activate physics
python scripts/day01_free_fall.py
```

或者用一键脚本：

```bash
./run_day.sh 00
./run_day.sh 01
./run_day.sh 05
```

输出图片会放在 `outputs/` 目录。

如果系统提示 `Permission denied`，先运行：

```bash
chmod +x run_day.sh
```

## 第一周脚本

- `day00_deep_learning_whole_picture.py`: 先看完整深度学习结构
- `day01_free_fall.py`: Python 基础 + 自由落体
- `day02_oscillator.py`: NumPy + 简谐振子
- `day03_linear_regression_numpy.py`: 用数据反推物理参数
- `day04_torch_tensor_autograd.py`: PyTorch 张量和自动求导
- `day05_fit_sine_torch.py`: 用神经网络拟合函数
- `day06_damped_oscillator_learning.py`: 用神经网络学习阻尼振动
- `day07_review_and_play.py`: 复盘、改参数、做小实验
