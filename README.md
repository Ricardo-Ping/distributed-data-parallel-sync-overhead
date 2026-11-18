# D1：数据并行中的计算与通信开销

项目研究模型规模、Batch Size、梯度累积和 Local SGD 同步间隔对训练性能的影响。E0—E5各有独立的代码、配置、测试和结果，原始数据统一放在`Data/`。训练、验证、测试和模型数值检查均使用CUDA，多GPU通信使用NCCL。

实验覆盖MNIST、FashionMNIST、CIFAR-10，训练种子为1234、1235、1236，每次训练20轮。共468次完整训练，包括匹配的单GPU对照；每轮验证，训练结束后测试并保存第20轮模型。

## 实验内容

| 实验 | 研究问题 | 完整运行数 | 说明与报告 |
|---|---|---:|---|
| E0 | 相同全局Batch下的单、双GPU数据并行 | 18 | [运行](E0/README.md) · [报告](E0/results/实验报告.md) |
| E1 | 三档MLP宽度对计算与通信比例的影响 | 54 | [运行](E1/README.md) · [报告](E1/results/实验报告.md) |
| E2 | 固定双GPU，每卡Batch为32—512 | 90 | [运行](E2/README.md) · [报告](E2/results/实验报告.md) |
| E3 | 固定全局Batch=256，改变Micro-Batch和累积次数 | 72 | [运行](E3/README.md) · [报告](E3/results/实验报告.md) |
| E4 | 同一累积窗口内，每Micro同步与末次同步的比较 | 135 | [运行](E4/README.md) · [报告](E4/results/实验报告.md) |
| E5 | Local SGD每1、2、4、8、16步平均参数 | 99 | [运行](E5/README.md) · [报告](E5/results/实验报告.md) |

## 目录与结果

```text
d1-gradient-sync/
├── Data/                   数据缓存与原始数据读取接口
├── E0/
│   ├── configs/            三个数据集的配置
│   ├── data.py             E0 的验证划分与采样
│   ├── model.py            MLP-M 与模型保存、加载
│   ├── train.py            单次训练的完整流程
│   ├── evaluate.py         验证和测试指标、模型独立评估
│   ├── distributed.py      CUDA/NCCL 与同步计时
│   ├── run.py              配对实验启动入口
│   ├── tests/              E0 的正确性检查
│   ├── results/            三种子记录、模型、图表和实验报告.md
│   └── README.md           使用方法
├── E1/                     模型规模实验，固定每卡batch，单/双GPU、三规模、三种子
│   ├── configs/、tests/     独立配置与正确性检查
│   ├── run.py、train.py     配对计划及训练流程
│   ├── check_results.py    原始记录复算与GPU复测
│   ├── plot_results.py     图表与机器统计表
│   └── results/            逐rank记录、模型、图表和实验报告.md
├── E2/                     实验A：固定双GPU，扫描每卡batch，累积1次
│   ├── single_gpu/         匹配全局batch的独立单卡实现
│   └── results/            45双卡+45单卡，逐步记录、模型、图表和实验报告.md
├── E3/                     实验B：固定双GPU和全局batch256，改变micro-batch与累积
│   ├── single_gpu/         匹配全局batch及K的单卡实现
│   └── results/            36双卡+36单卡，等价性记录、模型、图表和实验报告.md
├── E4/                     同K配对：每micro同步与窗口末同步，验证no_sync通信优化
│   ├── single_gpu/         45份基线分别匹配90种双卡运行
│   └── results/            90双卡+45单卡，逐步记录、图表与实验报告.md
└── E5/                     Local SGD：H=1/2/4/8/16，本地momentum、周期参数平均
    ├── single_gpu/         串行双副本基线、普通SGD任务参考及配对启动器
    └── results/            45双卡+54单卡，逐步/轮次、漂移/TTT/Profiler与报告
```

数据格式、缓存位置和下载方法见[Data](Data/README.md)。E2—E5共有180次单GPU训练：E4同K的两种双卡方式共用一份单卡基线；E5分别设置同算法串行双副本基线与普通SGD任务参考。

`.venv/`是本机依赖环境，可通过`uv sync --locked`重建；本机运行实验时保留。`.pytest_cache/`、`.ruff_cache/`和`__pycache__/`由测试、检查工具或Python自动生成，可以清理，不属于实验数据或结果。

每个实验的`results/step_metrics/`保存训练步耗时、实际样本数、样本吞吐、同步事件、逻辑载荷及原始文件位置，共2,868,120条rank或逻辑副本记录。E0记录rank 0；E1—E5记录各rank。E5分别记录本地SGD步和H步同步轮次。计时范围、预热和尾批处理见各实验报告。

报告保留三个种子的观测及均值，解释吞吐、更新耗时、同步占比、加速比和并行效率。负收益、异常高比值和运行波动按实测值呈现。E1固定每卡Batch，属于弱扩展；E2—E5按相同工作量匹配单GPU基线，不能直接混用两类扩展指标。

E2—E4的图1—4直接包含单GPU对照，E5的图1—6采用同算法的单GPU串行双副本基线。颜色区分执行方式，形状区分三个种子，粗线表示均值；单GPU跨卡通信指标为0。E4的同步策略比值和E5的两rank到达差、双GPU Profiler保持各自的适用范围。主图的数据与来源索引保存在各实验的`main_comparison.csv`和`main_comparison_sources.json`。

## 运行环境

项目运行于Linux和NVIDIA CUDA环境。完整实验需要两张可见GPU，每个分布式进程使用一张卡，通信后端为NCCL；单GPU训练只使用一张卡。模型训练、验证、测试与数值检查均在CUDA上执行。

实验环境如下，完整软件依赖由`uv.lock`锁定：

| 项目 | 实验使用的版本或配置 |
|---|---|
| 操作系统 | Ubuntu 24.04.3 LTS，Linux 6.8，x86_64 |
| GPU | 2 × NVIDIA GeForce RTX 4090，每卡总显存24,564 MiB（nvidia-smi） |
| NVIDIA驱动 | 595.91.07 |
| Python | 3.12.14；`.python-version`指定3.12 |
| PyTorch / torchvision | 2.15.0+cu132 / 0.29.2+cu132 |
| CUDA运行时 / NCCL | 13.2 / 2.30.7 |
| NumPy / pandas | 2.5.4 / 3.1.1 |
| Matplotlib / PyYAML | 3.11.2 / 6.0.3 |
| pytest | 9.2.0，用于代码检查 |

服务器需安装可支持上述CUDA运行时的NVIDIA驱动。绘图需要中文字体，程序支持Noto Sans CJK、Microsoft YaHei或SimHei；Ubuntu可安装`fonts-noto-cjk`。不同GPU、驱动和主机调度条件可能改变计时，报告中的性能数值对应上述实验环境。

## 依赖安装

以下命令均在包含`pyproject.toml`的项目根目录执行。若尚未安装uv，可使用[官方安装器](https://docs.astral.sh/uv/getting-started/installation/)：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装后重新打开终端，确认`uv --version`可以运行，再创建项目环境并安装锁定依赖：

```bash
uv python install 3.12
uv sync --locked

# Ubuntu绘图字体
sudo apt-get update
sudo apt-get install -y fonts-noto-cjk
```

`pyproject.toml`声明PyTorch、torchvision、NumPy、pandas、Matplotlib和PyYAML，开发依赖包含pytest。`uv sync --locked`按锁文件安装到`.venv/`，后续命令通过`uv run`使用该环境。

检查GPU、CUDA和NCCL：

```bash
nvidia-smi
uv run python - <<'PY'
import torch
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('GPU数量:', torch.cuda.device_count())
assert torch.cuda.is_available(), '需要CUDA GPU'
assert torch.cuda.device_count() >= 2, '完整对照实验需要两张GPU'
assert torch.distributed.is_nccl_available(), '需要支持NCCL的PyTorch'
PY
```

## 数据准备

```bash
# 首次运行允许下载三个数据集
uv run python -m Data.datasets --download

# 只检查已有缓存
uv run python -m Data.datasets
```

缓存统一保存在`Data/`。训练入口默认只读取已有数据，不在训练过程中下载。数据格式、预处理和各实验的尾批规则见[数据说明](Data/README.md)。

训练时，各实验会自动按固定种子生成训练集和验证集的分层划分，并按训练种子和轮次安排采样。下载缓存后即可启动训练。

## 训练

先用`--list`查看计划，不启动训练：

```bash
uv run python -m E0.run --list
uv run python -m E1.run --list
uv run python -m E2.run --list
uv run python -m E2.single_gpu.run --list
uv run python -m E5.single_gpu.paired_run --list
```

运行入口拒绝覆盖同名记录。下面使用独立的`results/reproduce/`保存一组完整复现结果，训练命令均包含每轮验证、20轮结束后的测试和模型保存。

```bash
export CUDA_VISIBLE_DEVICES=0,1

# E0、E1：入口包含单、双GPU配置
uv run python -m E0.run --output-dir E0/results/reproduce
uv run python -m E1.run --output-dir E1/results/reproduce

# E2—E4：分别执行双GPU配置与匹配的单GPU配置
for experiment in E2 E3 E4; do
    uv run python -m "${experiment}.run" --output-dir "${experiment}/results/reproduce"
    CUDA_VISIBLE_DEVICES=0 uv run python -m "${experiment}.single_gpu.run" --output-dir "${experiment}/results/reproduce/single_gpu"
done

# E5：配对入口包含双GPU、串行单GPU及普通SGD参考
uv run python -m E5.single_gpu.paired_run --dual-output-dir E5/results/reproduce --single-output-dir E5/results/reproduce/single_gpu
```

性能采集按任务顺序串行执行。E2—E5支持`--resume`跳过配置一致且完整的运行；未完成目录不会自动覆盖。`--dataset`、`--seed`等筛选参数用于单个组合调试，具体选项见各实验README或`--help`。

## 验证与测试

训练期间只使用验证集，最终测试使用第20轮模型。已有模型可以独立评估，例如：

```bash
uv run python -m E2.evaluate --checkpoint E2/results/reproduce/mnist/b128_k1/seed1234/checkpoint.pt --split validation --device cuda:0
uv run python -m E2.evaluate --checkpoint E2/results/reproduce/mnist/b128_k1/seed1234/checkpoint.pt --split test --device cuda:0
```

完整结果检查会复算记录并在GPU上独立复测模型。E2的示例为：

```bash
uv run python -m E2.check_results --results-dir E2/results/reproduce
uv run python -m E2.single_gpu.check_results --results-dir E2/results/reproduce/single_gpu --dual-results-dir E2/results/reproduce
```

E0—E5各自提供`evaluate`和`check_results`入口。主入口的`check_results --records-only`只检查记录和模型元数据。代码检查使用：

```bash
uv run pytest -q
```

模型计算检查使用CUDA，分布式检查需要双GPU和NCCL。

## 统计与绘图

E0、E1直接从各自完整结果生成图表：

```bash
uv run python -m E0.plot_results --results-dir E0/results/reproduce
uv run python -m E1.plot_results --results-dir E1/results/reproduce
```

E2—E5先计算匹配的加速比和并行效率，再绘制全部统计图：

```bash
for experiment in E2 E3 E4 E5; do
    uv run python -m "${experiment}.single_gpu.compare" --results-dir "${experiment}/results/reproduce/single_gpu" --dual-results-dir "${experiment}/results/reproduce" --output-dir "${experiment}/results/reproduce"
    uv run python -m "${experiment}.plot_results" --results-dir "${experiment}/results/reproduce"
    uv run python -m "${experiment}.single_gpu.plot_results" --results-dir "${experiment}/results/reproduce"
done
```

E5的图7需要独立Profiler记录，完成正式性能采集后单独运行，再绘图：

```bash
uv run python -m E5.profile_activity --results-dir E5/results/reproduce
uv run python -m E5.plot_results --results-dir E5/results/reproduce
```

图表保存在对应结果目录的`figs/`。E2、E3使用图1—7，E4使用图1—8，E5使用图1—11。
