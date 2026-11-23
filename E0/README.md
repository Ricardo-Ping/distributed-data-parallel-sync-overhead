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

完整对照需要两张可见 CUDA GPU 和 NCCL。先运行环境及数据准备脚本。版本与准备步骤见 [项目 README](../README.md#运行环境) · [数据说明](../Data/README.md)。

## 训练

以下命令在项目根目录执行。两种训练脚本各列出 9 次任务，每种卡数覆盖三个数据集、三个种子。运行时按顺序执行两条训练命令，共得到 18 次记录。

```bash
bash "sh/3.1.E0单卡训练.sh" --list
bash "sh/3.3.E0双卡训练.sh" --list

bash "sh/3.1.E0单卡训练.sh"
bash "sh/3.3.E0双卡训练.sh"
```

两种卡数共用默认目录 `E0/results/reproduce/`。可追加 `--dataset mnist --seed 1234` 选择代表配置，或用 `--output-dir` 指定独立目录。遇到已有的同名运行，入口会要求另选目录或先归档。归档的正式批次按种子交替卡数，复现其任务顺序可使用 `uv run --locked python -m E0.run --output-dir E0/results/reproduce_ordered`。

三份配置使用 20 轮、全局 batch 256、SGD momentum 0.9、10% 分层验证集和划分种子 1234。CIFAR-10 学习率为 0.05，另外两个数据集为 0.1。

## 验证与测试

每轮训练后，rank 0 在完整验证集上评估损失和准确率。终端还会打印该轮全局样本吞吐、平均更新耗时、平均同步耗时和同步占比。性能汇总使用实际样本数和 rank 0 的更新计时，包含预热步和尾批。最低验证损失所在轮次另作记录。第 20 轮验证完成后，训练入口评估官方测试集并保存该轮模型。

以下两条脚本默认分别读取 `E0/results/reproduce/` 中 MNIST、种子 1234 的单卡与双卡末轮模型，在 `cuda:0` 上复测 10,000 个测试样本。使用其他结果目录时，将实际检查点路径作为参数传入。路径见各数据集的 `final.csv` 中的 `checkpoint_path` 字段。

```bash
bash "sh/3.2.E0单卡测试.sh"
bash "sh/3.4.E0双卡测试.sh"
```

评估输出 JSON，包含样本数、损失、准确率、设备、数据集和模型轮数。需要重新评估验证集时，可用 `uv run --locked python -m E0.evaluate --checkpoint <检查点路径> --split validation --device cuda:0`。

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
bash "sh/3.5.E0绘图.sh"
```

绘图默认读取 `E0/results/reproduce/`。从现有归档重新绘图可传入 `--results-dir E0/results`。

该命令读取每个数据集的六次完整结果，共 18 次，输出图表及 `seed_results.csv`、`seed_summary.csv`、`epoch_summary.csv`。缺少任何一个种子时会报错。F1、F2、F4使用单卡与双卡并列柱状图，四组横坐标依次为1234、1235、1236及三种子均值，不绘制置信区间或误差条。F3逐轮曲线用黑线显示均值，三条有不同点形的彩线显示三个种子，灰色阴影是每轮独立计算的逐点95% t区间。该区间假设三个重复近似独立且正态，样本量较小，仅作探索性展示。

图内解释性文字以中文为主，Flatten、Linear、ReLU、Logits、SGD、NCCL AllReduce 等技术名称使用通用英文写法；数据集保留 MNIST、FashionMNIST、CIFAR-10 原名。绘图需要中文字体，推荐 Noto Sans CJK；Linux 系统可使用 `fonts-noto-cjk` 软件包。缺少可用中文字体时，程序会明确报错。图1展示训练吞吐，图2展示更新耗时、同步耗时和同步占比，图3(a)、图3(b)、图3(c)分别展示三个数据集的训练与验证过程，图4展示最终测试准确率；每张图的指标含义、单位与读图方式见报告。

## 正确性检查

```bash
# 代码回归，包括 GPU 梯度、尾批和末轮模型检查
uv run --locked pytest E0/tests -q

# 记录核验，随后在两张 GPU 上独立复测18个正式模型
uv run --locked python -m E0.check_results

# 只检查记录、划分和模型元数据
uv run --locked python -m E0.check_results --records-only
```

完整核验会重算各次运行的样本数、更新数、吞吐和同步占比，检查划分索引与所有 rank 的设备记录。测试集准确率要求与保存记录一致；浮点测试损失允许绝对误差1e-6或相对误差1e-5。完整复测写入 `validation.json`，只检查记录时写入 `records_validation.json`。

指标定义、各图含义及结果分析见[实验报告](results/实验报告.md)。

模型保存、加载和校验均使用CUDA。主机侧负责数据准备，运行元数据中的`host_threads`记录线程预算。18个末轮模型的独立GPU复测结果见[validation.json](results/validation.json)。
