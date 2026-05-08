# Day 00 深度学习完整结构逐行讲解

对应代码文件：

`scripts/day00_deep_learning_whole_picture.py`

这个脚本的目标不是做一个复杂模型，而是让你在一个最小例子里看到深度学习从头到尾的完整结构：

```text
数据 -> 模型 -> 预测 -> 损失 -> 反向传播 -> 参数更新 -> 重复训练 -> 可视化结果
```

你可以先把深度学习理解成一句话：

```text
用很多数据训练一个带有可调参数的函数，让它的预测越来越接近真实答案。
```

在这个例子里：

- 输入是 `x`
- 真实答案是 `y`
- 神经网络是 `model`
- 神经网络预测值是 `y_pred`
- 错误大小是 `loss`
- 调整参数的方法是 `optimizer`

## 完整代码结构

这个文件大致分成 6 个部分：

1. 导入工具
2. 准备输出目录和计算设备
3. 生成训练数据
4. 定义神经网络模型
5. 定义损失函数和优化器
6. 训练模型并画图

## 逐行讲解

### 第 1 行

```python
from pathlib import Path
```

这行代码从 Python 标准库 `pathlib` 里面导入 `Path`。

`Path` 用来处理文件路径。比如我们后面要把图片保存到 `outputs` 文件夹里，就会用到它。

你可以把 `Path("outputs")` 理解成：

```text
我要使用当前目录下面叫 outputs 的文件夹。
```

这不是深度学习本身的内容，只是为了保存输出图片。

### 第 3 行

```python
import matplotlib.pyplot as plt
```

这行导入画图库 `matplotlib` 里的 `pyplot` 模块，并给它取一个常用别名 `plt`。

后面所有画图操作，比如画散点图、画曲线、保存图片，都会通过 `plt` 完成。

在深度学习里，画图非常重要，因为你需要观察：

- 数据长什么样
- 模型拟合得怎么样
- loss 是否下降
- 训练有没有失败

### 第 4 行

```python
import torch
```

这行导入 PyTorch。

PyTorch 是我们用来做深度学习的核心库。它主要负责：

- 创建张量 `tensor`
- 搭建神经网络
- 自动求导
- 训练模型
- 使用 CPU 或 Apple 芯片加速

你可以把 `torch` 理解成深度学习里的“主工具箱”。

### 第 7 行

```python
out_dir = Path("outputs")
```

这行创建一个路径对象，表示 `outputs` 文件夹。

变量名 `out_dir` 是 `output directory` 的缩写，意思是“输出目录”。

后面保存图片时会用：

```python
out_dir / "day00_deep_learning_whole_picture.png"
```

意思是把图片保存到：

```text
outputs/day00_deep_learning_whole_picture.png
```

### 第 8 行

```python
out_dir.mkdir(exist_ok=True)
```

这行代码创建 `outputs` 文件夹。

`mkdir` 的意思是 `make directory`，也就是创建文件夹。

`exist_ok=True` 的意思是：

```text
如果这个文件夹已经存在，也不要报错。
```

这是为了保证脚本可以反复运行。

### 第 10 行

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

这行是在选择计算设备。

你的电脑是 Apple Silicon 芯片，所以 PyTorch 可以使用 `mps` 加速。`mps` 可以简单理解为 Mac 上的 GPU 加速方式。

这行代码的逻辑是：

```text
如果 mps 可用，就用 mps；
否则就用 cpu。
```

写成普通中文就是：

```python
如果 torch.backends.mps.is_available() 为 True:
    device = "mps"
否则:
    device = "cpu"
```

在深度学习里，模型和数据必须放在同一个设备上。比如数据在 `mps`，模型也要在 `mps`。

### 第 11 行

```python
torch.manual_seed(7)
```

这行设置随机种子。

深度学习里有很多随机过程，例如：

- 神经网络参数一开始是随机初始化的
- 噪声数据是随机生成的
- 有些训练过程也可能带随机性

设置 `manual_seed(7)` 之后，每次运行时随机结果会尽量保持一致。

这对学习很重要，因为你不希望每次运行出来的图差别太大。

数字 `7` 没有什么特殊含义，换成 `0`、`42` 也可以。

### 第 13 行

```python
# 1. Data: pretend these are experimental measurements from an unknown law.
```

这是注释。

Python 看到 `#` 开头的内容不会执行，它只是写给人看的说明。

这句的意思是：

```text
第 1 部分：数据。我们假装这些数据来自某个未知物理规律的实验测量。
```

这很重要，因为深度学习通常从数据开始。

### 第 14 行

```python
x = torch.linspace(-3, 3, 120).reshape(-1, 1).to(device)
```

这一行做了三件事：

第一步：

```python
torch.linspace(-3, 3, 120)
```

生成从 `-3` 到 `3` 的 `120` 个等间距数字。

你可以把它想象成实验中取了 120 个输入点：

```text
x = -3, -2.95, -2.90, ..., 3
```

第二步：

```python
.reshape(-1, 1)
```

把数据形状改成 `120 行 1 列`。

深度学习模型通常希望输入是二维的：

```text
[样本数量, 每个样本的特征数量]
```

这里就是：

```text
[120, 1]
```

意思是：

```text
一共有 120 个样本，每个样本只有 1 个输入特征 x。
```

第三步：

```python
.to(device)
```

把数据放到前面选择好的设备上，也就是 `mps` 或 `cpu`。

这一行在深度学习结构里对应：

```text
准备输入数据 x
```

### 第 15 行

```python
y_true = 0.5 * x**2 - 1.0 * x + 0.3
```

这一行定义真实规律。

它是一个二次函数：

```text
y = 0.5x² - 1.0x + 0.3
```

在真实科研里，这个规律可能来自实验或物理方程。但在这个入门例子中，我们自己用公式生成它。

变量名 `y_true` 的意思是“真实值”。

这行代码在深度学习结构中对应：

```text
准备正确答案 y_true
```

注意：神经网络并不知道这个公式。它后面只会看到数据点，然后自己去学。

### 第 16 行

```python
y = y_true + 0.25 * torch.randn_like(y_true)
```

这一行给真实数据加噪声。

先看：

```python
torch.randn_like(y_true)
```

它会生成一个和 `y_true` 形状一样的随机噪声张量。

再看：

```python
0.25 * torch.randn_like(y_true)
```

把噪声缩小到一定幅度。`0.25` 越大，噪声越强；越小，数据越干净。

最终：

```python
y = y_true + 噪声
```

意思是我们模拟真实实验数据：

```text
真实规律 + 测量误差 = 观测数据
```

在深度学习里，模型通常学习的是带噪声的真实世界数据，而不是完美公式。

### 第 18 行

```python
# 2. Model: a tiny neural network that maps x -> y.
```

这是注释。

意思是：

```text
第 2 部分：模型。我们定义一个小神经网络，让它学习从 x 到 y 的映射。
```

深度学习里的模型可以先理解成一个函数：

```text
model(x) = y_pred
```

输入 `x`，输出预测值 `y_pred`。

### 第 19 行

```python
model = torch.nn.Sequential(
```

这一行开始定义神经网络模型。

`torch.nn` 是 PyTorch 里专门用来搭建神经网络的模块。

`Sequential` 的意思是“按顺序堆起来”。

它会把后面的层按照顺序连接起来：

```text
输入 -> 第一层 -> 激活函数 -> 第二层 -> 激活函数 -> 输出层
```

### 第 20 行

```python
    torch.nn.Linear(1, 16),
```

这是一层全连接层，也叫线性层。

`Linear(1, 16)` 的意思是：

```text
输入维度是 1，输出维度是 16。
```

因为我们的输入只有一个数字 `x`，所以输入维度是 1。

输出 16 维可以理解成：模型把一个简单的输入数字变成 16 个内部特征。

这一层里面有可训练参数，也就是权重和偏置。训练深度学习模型，本质上就是不断修改这些参数。

### 第 21 行

```python
    torch.nn.Tanh(),
```

这是激活函数。

如果神经网络只有线性层，那么不管堆多少层，本质上还是线性变换，很难拟合复杂曲线。

`Tanh` 会引入非线性，使模型可以学习弯曲的函数，比如二次函数、正弦函数、物理轨迹等。

你现在可以记住：

```text
Linear 负责可调参数，Tanh 负责让模型有非线性表达能力。
```

### 第 22 行

```python
    torch.nn.Linear(16, 16),
```

这是第二个线性层。

`Linear(16, 16)` 的意思是：

```text
输入 16 个特征，输出 16 个特征。
```

这一层继续加工前一层提取出来的特征。

层数越多、特征数越多，模型表达能力通常越强，但也更容易过拟合，训练也可能更不稳定。

### 第 23 行

```python
    torch.nn.Tanh(),
```

这是第二个激活函数。

它继续给网络加入非线性能力。

你可以把目前网络想象成：

```text
x -> 变成 16 个特征 -> 非线性变换 -> 再加工 -> 再非线性变换
```

### 第 24 行

```python
    torch.nn.Linear(16, 1),
```

这是输出层。

`Linear(16, 1)` 的意思是：

```text
输入 16 个内部特征，输出 1 个数字。
```

因为我们最终只想预测一个 `y` 值，所以输出维度是 1。

整个模型的输入输出关系是：

```text
输入一个 x -> 输出一个预测 y_pred
```

### 第 25 行

```python
).to(device)
```

这一行结束模型定义，并把模型移动到计算设备上。

前面的 `x` 和 `y` 已经被放到了 `device` 上，所以模型也必须放到同一个设备。

如果数据在 `mps`，模型在 `cpu`，PyTorch 会报错。

这一行在深度学习结构里对应：

```text
准备模型 model
```

### 第 27 行

```python
# 3. Loss: how wrong the prediction is.
```

这是注释。

意思是：

```text
第 3 部分：损失函数。它用来衡量模型预测错了多少。
```

深度学习训练的核心目标就是：

```text
让 loss 越来越小。
```

### 第 28 行

```python
loss_fn = torch.nn.MSELoss()
```

这一行定义损失函数。

`MSE` 是 `Mean Squared Error`，中文叫“均方误差”。

它大致计算：

```text
预测值和真实值的差距平方，再求平均。
```

公式可以理解成：

```text
loss = 平均值((y_pred - y)²)
```

在这个任务里，我们希望神经网络输出的曲线尽量接近带噪声数据，所以用 MSE 很合适。

### 第 30 行

```python
# 4. Optimizer: how the model changes its parameters.
```

这是注释。

意思是：

```text
第 4 部分：优化器。它决定模型参数怎么更新。
```

如果损失函数告诉我们“现在错了多少”，优化器就负责“下一步怎么改参数”。

### 第 31 行

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
```

这一行定义优化器。

`torch.optim.Adam` 是一种常用优化算法。初学阶段直接用 Adam 很合适。

看两个关键部分：

```python
model.parameters()
```

意思是把模型里所有可训练参数交给优化器。

比如每个 `Linear` 层里的权重和偏置，都会被包含进去。

再看：

```python
lr=0.03
```

`lr` 是 `learning rate`，中文叫学习率。

学习率决定每次参数更新迈多大一步：

- 太小：训练很慢
- 太大：训练可能不稳定，甚至越学越差

这一行在深度学习结构里对应：

```text
准备优化器 optimizer
```

### 第 33 行

```python
loss_history = []
```

这一行创建一个空列表，用来记录每一轮训练的 loss。

后面每训练一轮，就把当前 loss 加进去。

这样最后就可以画出 loss 曲线，观察训练过程是否真的在进步。

### 第 35 行

```python
# 5. Training loop: predict -> measure error -> compute gradients -> update.
```

这是注释。

意思是：

```text
第 5 部分：训练循环。
```

训练循环是深度学习最核心的结构：

```text
预测 -> 计算错误 -> 计算梯度 -> 更新参数
```

这个过程会重复很多次。

### 第 36 行

```python
for epoch in range(800):
```

这一行开始训练循环。

`epoch` 可以理解成训练轮数。

`range(800)` 表示从 `0` 到 `799`，一共训练 800 轮。

每一轮都会完整执行下面缩进的代码：

```python
y_pred = model(x)
loss = loss_fn(y_pred, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 第 37 行

```python
    y_pred = model(x)
```

这一行让模型进行预测。

输入是 `x`，输出是 `y_pred`。

你可以理解成：

```text
当前这个神经网络，根据当前参数，对所有 x 做出预测。
```

一开始模型参数是随机的，所以预测通常很差。

随着训练进行，模型参数被不断更新，预测会越来越接近 `y`。

### 第 38 行

```python
    loss = loss_fn(y_pred, y)
```

这一行计算损失。

它比较：

- `y_pred`：模型预测值
- `y`：带噪声的观测数据

然后得到一个数字 `loss`。

这个数字越大，说明预测越差；越小，说明预测越接近数据。

### 第 40 行

```python
    optimizer.zero_grad()
```

这一行清空上一轮的梯度。

PyTorch 默认会累加梯度。如果不清空，上一轮的梯度会和这一轮混在一起。

所以每次反向传播之前都要先写：

```python
optimizer.zero_grad()
```

你可以把它理解成：

```text
先把上一轮的计算痕迹擦掉，准备重新计算这一轮该怎么改参数。
```

### 第 41 行

```python
    loss.backward()
```

这是深度学习最关键的一行之一。

`backward()` 会根据当前 loss，自动计算模型中每个参数的梯度。

梯度可以先理解成：

```text
为了让 loss 变小，每个参数应该往哪个方向调整。
```

你不需要手动推导每个权重的偏导数，PyTorch 会自动完成。

这就是 PyTorch 的自动求导机制。

### 第 42 行

```python
    optimizer.step()
```

这一行真正更新模型参数。

前一行 `loss.backward()` 只是计算“应该怎么改”。

这一行 `optimizer.step()` 才是真正执行“开始改参数”。

所以训练中最核心的三连是：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

它们的意思分别是：

```text
清空旧梯度 -> 计算新梯度 -> 更新模型参数
```

### 第 44 行

```python
    loss_history.append(loss.item())
```

这一行把当前这一轮的 loss 保存到列表里。

`loss` 本身是 PyTorch 张量。

`loss.item()` 会把它转成普通 Python 数字。

`append` 的意思是把这个数字追加到列表末尾。

最后 `loss_history` 里会有 800 个数字，对应 800 轮训练的 loss。

### 第 45 行

```python
    if epoch % 100 == 0:
```

这一行判断当前轮数是不是 100 的倍数。

`%` 是取余数。

比如：

```text
200 % 100 = 0
250 % 100 = 50
```

所以这行的意思是：

```text
每隔 100 轮，打印一次训练进度。
```

### 第 46 行

```python
        print(f"epoch={epoch:03d}, loss={loss.item():.6f}")
```

这一行打印当前训练轮数和 loss。

`f"..."` 是 Python 的格式化字符串。

`{epoch:03d}` 的意思是把 epoch 显示成至少 3 位整数。

例如：

```text
0 -> 000
5 -> 005
100 -> 100
```

`{loss.item():.6f}` 的意思是把 loss 显示成保留 6 位小数。

打印结果类似：

```text
epoch=000, loss=7.680798
epoch=100, loss=0.079099
```

如果 loss 从大变小，说明模型在学习。

### 第 48 行

```python
# 6. Prediction: use the trained model.
```

这是注释。

意思是：

```text
第 6 部分：训练结束后，使用训练好的模型做预测。
```

训练阶段是为了更新参数。训练结束后，我们通常要用模型做推断或预测。

### 第 49 行

```python
with torch.no_grad():
```

这行表示进入“不计算梯度”的模式。

训练时需要梯度，因为要更新参数。

但预测时不需要梯度，只需要输出结果。

使用 `torch.no_grad()` 有两个好处：

- 更省内存
- 更快

### 第 50 行

```python
    y_learned = model(x)
```

这一行用训练好的模型重新预测所有 `x` 对应的 `y`。

变量名 `y_learned` 的意思是：

```text
神经网络学出来的 y 值。
```

后面会把它和真实规律 `y_true`、带噪声数据 `y` 画在同一张图里。

### 第 52 行

```python
print("\n一个深度学习任务的完整结构：")
```

这一行打印一句提示。

`\n` 表示换行，让输出更清楚。

### 第 53 行

```python
print("数据 data -> 模型 model -> 预测 prediction -> 损失 loss -> 反向传播 backward -> 优化器 optimizer -> 重复训练")
```

这一行打印深度学习的完整流程。

这句话非常重要，你可以背下来：

```text
数据 -> 模型 -> 预测 -> 损失 -> 反向传播 -> 优化器 -> 重复训练
```

几乎所有深度学习任务，无论是图像识别、语音识别、物理建模、语言模型，底层训练结构都离不开这条链。

### 第 55 行

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
```

这一行创建画布和子图。

`plt.subplots(1, 2)` 的意思是：

```text
创建 1 行 2 列的图。
```

也就是左右两个图。

`figsize=(10, 4)` 表示整张图宽 10、高 4。

返回的：

- `fig`：整张图
- `axes`：两个子图对象

后面：

- `axes[0]` 表示左边那张图
- `axes[1]` 表示右边那张图

### 第 57 行

```python
axes[0].scatter(x.cpu(), y.cpu(), s=18, label="noisy data")
```

这一行在左图画散点图。

`scatter` 表示散点。

画的是：

- 横坐标：`x`
- 纵坐标：`y`

也就是带噪声的观测数据。

`s=18` 表示点的大小。

`label="noisy data"` 表示图例名称。

这里使用 `.cpu()` 是因为 Matplotlib 画图通常需要 CPU 上的数据。如果数据在 `mps` 上，需要先移动回 CPU。

### 第 58 行

```python
axes[0].plot(x.cpu(), y_true.cpu(), label="true law")
```

这一行在左图画真实规律曲线。

画的是：

```text
x 对应的 y_true
```

也就是没有噪声的真实二次函数。

这条线是我们用来对照神经网络学得好不好的标准。

### 第 59 行

```python
axes[0].plot(x.cpu(), y_learned.cpu(), "--", label="neural network")
```

这一行画神经网络学出来的曲线。

`"--"` 表示用虚线画。

如果这条虚线接近 `true law`，说明神经网络学到了背后的规律。

这一行就是最终可视化模型效果。

### 第 60 行

```python
axes[0].set_title("Learn a Hidden Physical Law")
```

这一行设置左图标题。

标题意思是：

```text
学习一个隐藏的物理规律
```

这个例子虽然只是二次函数，但思想和物理建模类似：

```text
从观测数据中学习背后规律。
```

### 第 61 行

```python
axes[0].set_xlabel("x")
```

设置左图横轴名称为 `x`。

### 第 62 行

```python
axes[0].set_ylabel("y")
```

设置左图纵轴名称为 `y`。

### 第 63 行

```python
axes[0].grid(True)
```

给左图添加网格线。

网格线可以帮助你更容易看出曲线和数据点的位置关系。

### 第 64 行

```python
axes[0].legend()
```

显示左图图例。

图例会告诉你：

- 哪些点是 noisy data
- 哪条线是 true law
- 哪条虚线是 neural network

### 第 66 行

```python
axes[1].plot(loss_history)
```

这一行在右图画 loss 曲线。

横轴默认是训练轮数，纵轴是 loss。

这张图非常重要，因为它显示模型训练过程中有没有进步。

通常我们希望看到：

```text
loss 从大变小，然后逐渐稳定。
```

如果 loss 不下降，说明模型没有学好，可能原因包括：

- 学习率不合适
- 模型太小
- 数据有问题
- 训练轮数不够

### 第 67 行

```python
axes[1].set_title("Training Loss")
```

设置右图标题为 `Training Loss`。

意思是训练损失。

### 第 68 行

```python
axes[1].set_xlabel("epoch")
```

设置右图横轴名称为 `epoch`。

`epoch` 表示训练轮数。

### 第 69 行

```python
axes[1].set_ylabel("MSE loss")
```

设置右图纵轴名称为 `MSE loss`。

也就是均方误差损失。

### 第 70 行

```python
axes[1].grid(True)
```

给右图添加网格线，方便观察 loss 下降趋势。

### 第 72 行

```python
plt.tight_layout()
```

自动调整图中各元素的位置。

如果没有这行，标题、坐标轴文字、图例可能会挤在一起。

### 第 73 行

```python
plt.savefig(out_dir / "day00_deep_learning_whole_picture.png", dpi=160)
```

保存图片。

保存路径是：

```text
outputs/day00_deep_learning_whole_picture.png
```

`dpi=160` 表示图片清晰度。

数字越大，图片越清晰，但文件也会更大。

### 第 74 行

```python
plt.show()
```

显示图片。

如果你在普通终端中运行，可能会弹出图片窗口；如果在某些非交互环境里运行，图片窗口可能不会显示，但第 73 行已经把图片保存下来了。

## 最核心的 6 行

如果你只能记住这份代码的一小部分，请先记住训练循环里的这 6 行：

```python
y_pred = model(x)
loss = loss_fn(y_pred, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

它们就是深度学习训练的心脏：

```text
模型预测 -> 计算错误 -> 清空旧梯度 -> 计算新梯度 -> 更新参数
```

## 这份代码和物理研究有什么关系

这个例子里，我们手动制造了一个规律：

```text
y = 0.5x² - x + 0.3
```

然后假装我们不知道这个规律，只知道带噪声的数据点。

神经网络做的事情就是：

```text
从数据点中学出接近真实规律的函数。
```

这和物理研究中的很多任务是相似的：

- 从实验数据中拟合规律
- 从轨迹数据中预测未来状态
- 从测量结果中反推物理参数
- 用神经网络近似复杂方程的解
- 把物理方程写进损失函数，也就是 PINN

## 今天你应该做的 3 个小实验

### 实验 1：改变噪声大小

找到这一行：

```python
y = y_true + 0.25 * torch.randn_like(y_true)
```

把 `0.25` 改成：

```python
0.05
```

再改成：

```python
1.0
```

观察：数据越吵，神经网络是不是越难学到平滑规律？

### 实验 2：改变模型大小

找到这些行：

```python
torch.nn.Linear(1, 16)
torch.nn.Linear(16, 16)
torch.nn.Linear(16, 1)
```

把 `16` 改成 `4`，再改成 `64`。

观察：模型太小和模型较大时，拟合曲线有什么不同？

### 实验 3：改变学习率

找到这一行：

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
```

把 `0.03` 改成：

```python
0.001
```

再改成：

```python
0.1
```

观察：loss 下降速度和稳定性有什么变化？

## 一句话总结

这份代码展示了深度学习最小完整闭环：

```text
生成数据，用神经网络预测，用 loss 衡量错误，用 backward 自动计算该怎么改，用 optimizer 更新参数，重复训练，最后画图检查结果。
```

