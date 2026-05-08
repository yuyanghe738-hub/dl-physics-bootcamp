# 30 天快速上手计划

原则：先动手，后补理论。每天 45-90 分钟即可。

## 第 1 周：Python 变成物理计算器

Day 1：自由落体  
任务：跑 `day01_free_fall.py`，改初速度 `v0` 和重力加速度 `g`。  
你要理解：变量、列表、循环、画图。

Day 2：简谐振子  
任务：跑 `day02_oscillator.py`，改质量 `m`、弹簧常数 `k`。  
你要理解：数组、时间步、数值模拟。

Day 3：线性回归反推物理参数  
任务：跑 `day03_linear_regression_numpy.py`，看程序如何从带噪声数据恢复斜率。  
你要理解：数据、模型、误差。

Day 4：PyTorch 张量与自动求导  
任务：跑 `day04_torch_tensor_autograd.py`。  
你要理解：tensor、requires_grad、backward。

Day 5：神经网络拟合 `sin(x)`  
任务：跑 `day05_fit_sine_torch.py`，改隐藏层大小和训练轮数。  
你要理解：模型、loss、optimizer、训练循环。

Day 6：神经网络学习阻尼振动  
任务：跑 `day06_damped_oscillator_learning.py`。  
你要理解：深度学习如何拟合物理曲线。

Day 7：复盘  
任务：跑 `day07_review_and_play.py`，自由改参数，保存三张你觉得有意思的图。  
你要理解：学会观察训练结果，而不是只看代码是否报错。

## 第 2 周：深度学习基本功

Day 8：手写一个训练循环，不复制代码。  
Day 9：学习 train/test split。  
Day 10：观察过拟合：训练点少一点，网络大一点。  
Day 11：加入噪声，看看模型如何失败。  
Day 12：用神经网络预测振子未来位置。  
Day 13：从轨迹反推阻尼系数。  
Day 14：整理一页学习笔记。

## 第 3 周：图像和经典网络

Day 15：MNIST 手写数字分类。  
Day 16：理解 batch 和 dataloader。  
Day 17：训练一个小 CNN。  
Day 18：看错分样本。  
Day 19：调学习率。  
Day 20：保存和加载模型。  
Day 21：复盘经典训练流程。

## 第 4 周：走向物理深度学习

Day 22：用网络拟合一阶 ODE 的解。  
Day 23：把微分方程残差写进 loss。  
Day 24：PINN 解简单 ODE。  
Day 25：PINN 加边界条件。  
Day 26：热方程的数值解。  
Day 27：神经网络拟合热方程数据。  
Day 28：比较数值解和神经网络解。  
Day 29：选一个你自己的物理问题。  
Day 30：做一个小报告：问题、数据、模型、结果图。

