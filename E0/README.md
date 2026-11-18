# E0：单 GPU 与双 GPU 基线

E0 在全局 batch 相同的条件下，比较单 GPU 和双 GPU 的训练结果与耗时。使用 MNIST、FashionMNIST、CIFAR-10，模型为 MLP-M，每次固定训练20轮。

完整报告位于 [results/实验报告.md](results/实验报告.md)，包含模型结构图、单卡/双卡训练对照表与流程图、指标定义、中文图表及逐图说明、18次逐种子结果，以及训练、验证和测试命令。这里说明代码入口。

单卡由一份完整模型处理256个样本后直接更新；双卡每张卡各保留完整模型、各处理128个样本，经 NCCL 平均梯度后分别更新。两组全局批量均为256、训练20轮、使用相同学习率。两组验证和最终测试都由第一个进程在单张 GPU 上完成。

## 代码顺序

1. `run.py` 读取 `configs/`，按种子安排单卡、双卡配对运行。
2. `data.py` 调用顶层 `Data/datasets.py` 读取图像，从官方训练集分层留出验证集，创建训练采样器。
3. `train.py` 建立模型和 SGD，依次执行20轮训练与验证；最后测试并保存末轮模型。
4. `evaluate.py` 负责验证和测试指标，也可以独立加载保存模型进行评估。

`model.py`定义模型与检查点格式，`distributed.py`负责设备、NCCL和同步计时，`metrics.py`定义CSV字段。E0只使用本目录的训练组件和顶层Data接口。

## 运行环境

需要支持 CUDA/NCCL 的 PyTorch。完整对照需要两张可见 GPU，每个进程独占一张卡。实验使用的软硬件版本见项目的[运行环境](../README.md#运行环境)，依赖、中文字体安装和GPU检查见[依赖安装](../README.md#依赖安装)。数据缓存默认是项目根目录的 `Data/`，路径按代码位置确定；首次运行先按[数据说明](../Data/README.md)准备缓存。

下面的命令在项目根目录执行：

```bash
uv sync --locked
nvidia-smi
uv run python -m E0.run --list
```

## 训练

`results/mnist/`、`results/fashionmnist/`、`results/cifar10/`保存完整结果。运行入口拒绝向相同`run_id`追加记录，独立复现时指定输出位置：

```bash
# 三个数据集，共18次训练
NCCL_DEBUG=INFO uv run python -m E0.run --output-dir E0/results/new_run

# 单个数据集的六次训练
uv run python -m E0.run --dataset mnist --output-dir E0/results/mnist_new_run

# 仅单卡，仍使用三个种子
uv run python -m E0.run --dataset mnist --world-size 1 --output-dir E0/results/single_check
```

也可以进入 `E0/` 后执行 `uv run python run.py --list`。完整实验按 `1234: 单→双`、`1235: 双→单`、`1236: 单→双` 的顺序串行运行。

默认不传 `--seed` 就会使用全部三个种子；只调试某个种子时才传 `--seed 1234`。正式三种子图表要求每个数据集都有六次完整结果。

配置使用平铺字段。三个 YAML 中除数据集和学习率外的协议一致：20轮、全局 batch 256、SGD momentum 0.9、10%分层验证集、划分种子1234。CIFAR-10 学习率为0.05，另外两个为0.1。

## 验证与测试

每轮训练结束后，仅 rank 0 在完整验证集上计算损失和准确率。双卡训练的另一个进程等待验证完成后再进入下一轮。验证损失最优轮只作记录，不决定停止或恢复模型。

20轮结束后，使用当前模型评估一次完整官方测试集。保存模型与这次测试对应。`evaluate_model()` 使用 `eval()` 和 `torch.inference_mode()`，按实际样本数统计损失与准确率，并恢复调用前的模型模式。

独立评估已有模型：

```bash
# 第20轮模型的完整测试集
uv run python -m E0.evaluate --checkpoint E0/results/mnist/checkpoints/e0_baseline_k1_ws1_mM_b256_r0_epoch20.pt --device cuda:0

# 同一个模型在原验证子集上评估
uv run python -m E0.evaluate --checkpoint E0/results/mnist/checkpoints/e0_baseline_k1_ws1_mM_b256_r0_epoch20.pt --split validation --device cuda:1
```

评估输出 JSON，包含样本数、损失、准确率、设备、数据集和模型轮数。训练、验证和测试均要求 CUDA。

## 结果位置

每个数据集各有一个目录：

```text
results/mnist/                 另有 fashionmnist/、cifar10/
├── dataset.json               划分规模、类别计数与索引哈希
├── split_indices.npz          训练、验证在官方训练集中的索引
├── runs.csv                   每次参数更新的分段耗时和同步量
├── epochs.csv                 每轮训练损失和验证结果
├── final.csv                  末轮模型测试、全程统计与模型路径
├── checkpoints/               三种子、单/双卡的六个模型
└── runtime/                   各 rank 的实际 GPU 与后端
```

图表位于[results/figs](results/figs/)，运行环境在[environment.json](results/environment.json)。统计表保留三个种子的均值、样本标准差、最小—最大范围、标准误和95% t均值置信限，`run_id`用于关联原始记录。

重新绘制逐种子性能、20轮训练/验证及末轮测试图：

```bash
uv run python -m E0.plot_results
```

该命令读取三个数据集的完整六次结果，输出图表及 `seed_results.csv`、`seed_summary.csv`、`epoch_summary.csv`。缺少任何一个种子时会报错。F1、F2、F4使用单卡与双卡并列柱状图，四组横坐标依次为1234、1235、1236及三种子均值，不绘制置信区间或误差条。F3逐轮曲线用黑线显示均值，三条有不同点形的彩线显示三个种子，灰色阴影是每轮独立计算的逐点95% t区间。该区间假设三个重复近似独立且正态，样本量较小，仅作探索性展示。

图内解释性文字以中文为主，Flatten、Linear、ReLU、Logits、SGD、NCCL AllReduce 等技术名称使用通用英文写法；数据集保留 MNIST、FashionMNIST、CIFAR-10 原名。绘图需要中文字体，推荐 Noto Sans CJK；Linux 系统可使用 `fonts-noto-cjk` 软件包。缺少可用中文字体时，程序会明确报错。图1展示训练吞吐，图2展示更新耗时、同步耗时和同步占比，图3(a)、图3(b)、图3(c)分别展示三个数据集的训练与验证过程，图4展示最终测试准确率；每张图的指标含义、单位与读图方式见报告。

## 正确性检查

```bash
# 代码回归，包括 GPU 梯度、尾批和末轮模型检查
uv run pytest E0/tests -q

# 记录核验，随后在两张 GPU 上独立复测18个正式模型
uv run python -m E0.check_results

# 只检查记录、划分和模型元数据
uv run python -m E0.check_results --records-only
```

完整核验会重算各次运行的样本数、更新数、吞吐和同步占比，检查划分索引与所有 rank 的设备记录。测试集准确率要求与保存记录一致；浮点测试损失允许绝对误差1e-6或相对误差1e-5。完整复测写入 `validation.json`，只检查记录时写入 `records_validation.json`。

指标定义、各图含义及结果分析见[实验报告](results/实验报告.md)。

模型保存、加载和校验均使用CUDA。主机侧负责数据准备，运行元数据中的`host_threads`记录线程预算。18个末轮模型的独立GPU复测结果见[validation.json](results/validation.json)。
