# D1：数据并行中的计算与通信开销

项目研究模型规模、Batch Size 和梯度累积对训练性能的影响。E0—E3各有独立的代码、配置、测试和结果，原始数据统一放在`Data/`。训练、验证、测试和模型数值检查均使用CUDA，多GPU通信使用NCCL。

实验覆盖MNIST、FashionMNIST、CIFAR-10，训练种子为1234、1235、1236，每次训练20轮。正式归档共234次完整训练，包括匹配的单GPU对照。每轮验证，训练结束后测试并保存第20轮模型。

## 实验内容

| 实验 | 研究问题 | 完整运行数 | 说明与报告 |
|---|---|---:|---|
| E0 | 相同全局Batch下的单、双GPU数据并行 | 18 | [运行](E0/README.md) · [报告](E0/results/实验报告.md) |
| E1 | 三档MLP宽度对计算与通信比例的影响 | 54 | [运行](E1/README.md) · [报告](E1/results/实验报告.md) |
| E2 | 固定双GPU，每卡Batch为32—512 | 90 | [运行](E2/README.md) · [报告](E2/results/实验报告.md) |
| E3 | 固定全局Batch=256，改变Micro-Batch和累积次数 | 72 | [运行](E3/README.md) · [报告](E3/results/实验报告.md) |

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
├── sh/                     环境、数据、单卡与双卡训练和测试、核验与绘图
└── training_common/        训练时共享的逐轮性能统计
```

数据格式、缓存位置和下载方法见[Data](Data/README.md)。E2、E3共有81次匹配的单GPU训练。

`.venv/`是本机依赖环境，可通过`uv sync --locked`重建；本机运行实验时保留。`.pytest_cache/`、`.ruff_cache/`和`__pycache__/`由测试、检查工具或Python自动生成，可以清理，不属于实验数据或结果。

每个实验的`results/step_metrics/`保存训练步耗时、实际样本数、样本吞吐、同步事件、逻辑载荷及原始文件位置，共1,765,740条逐rank记录。E0记录rank 0，E1—E3记录各rank。计时边界、预热和尾批处理见各实验报告。

报告保留三个种子的观测及均值，解释吞吐、更新耗时、同步占比、加速比和并行效率。负收益、异常高比值和运行波动按实测值呈现。E1固定每卡Batch，属于弱扩展。E2、E3按相同工作量匹配单GPU基线。

E2、E3的图1—4包含匹配的单GPU对照。颜色区分执行方式，形状区分三个种子，粗线表示均值。单GPU跨卡通信指标为0。主图的数据与来源索引保存在各自的`main_comparison.csv`和`main_comparison_sources.json`。

## 运行环境

项目运行于Linux和NVIDIA CUDA环境。完整实验需要两张可见GPU，每个分布式进程使用一张卡，通信后端为NCCL；单GPU训练只使用一张卡。模型训练、验证、测试与数值检查均在CUDA上执行。

实验环境如下，完整软件依赖由 `uv.lock` 锁定。

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

以下命令在包含 `pyproject.toml` 的项目根目录执行。完整环境检查需要 NVIDIA 驱动、两张可见 CUDA GPU 和中文绘图字体。Ubuntu 可用 `sudo apt-get install -y fonts-noto-cjk` 安装字体。

```bash
bash "sh/1.环境与依赖准备.sh"
```

脚本检查 `nvidia-smi` 和 GPU 拓扑，按需安装 uv 与 Python 3.12，执行 `uv sync --locked`，再核对 CUDA、GPU 数量、NCCL、P2P 和中文字体。环境检查默认选用 GPU 0、1。命令行安装的 uv 位于 `~/.local/bin`，后续直接运行 `uv` 前可在当前终端设置 `export PATH="$PATH:$HOME/.local/bin"`。完整软件依赖以 `pyproject.toml` 和 `uv.lock` 为准。

## 数据准备

```bash
bash "sh/2.数据集准备与划分.sh"
```

脚本准备或核对 MNIST、FashionMNIST、CIFAR-10 的官方缓存。各实验从 `Data/` 读取缓存，并在训练时按配置生成划分及采样索引。数据格式和划分规则见 [数据说明](Data/README.md)。

## 训练

E0—E3 均有独立的单卡训练、单卡测试、双卡训练和双卡测试入口。单卡训练绑定 GPU 0，双卡训练绑定 GPU 0、1。训练脚本传入 `--list` 只列出该卡数的运行计划。

| 实验 | 单卡训练 | 单卡测试 | 双卡训练 | 双卡测试 |
|---|---|---|---|---|
| E0 | [3.1](sh/3.1.E0单卡训练.sh) | [3.2](sh/3.2.E0单卡测试.sh) | [3.3](sh/3.3.E0双卡训练.sh) | [3.4](sh/3.4.E0双卡测试.sh) |
| E1 | [4.2.1](sh/4.2.1.E1单卡训练.sh) | [4.3.1](sh/4.3.1.E1单卡测试.sh) | [4.2.2](sh/4.2.2.E1双卡训练.sh) | [4.3.2](sh/4.3.2.E1双卡测试.sh) |
| E2 | [5.1](sh/5.1.E2单卡训练.sh) | [5.2](sh/5.2.E2单卡测试.sh) | [5.3](sh/5.3.E2双卡训练.sh) | [5.4](sh/5.4.E2双卡测试.sh) |
| E3 | [6.1](sh/6.1.E3单卡训练.sh) | [6.2](sh/6.2.E3单卡测试.sh) | [6.3](sh/6.3.E3双卡训练.sh) | [6.4](sh/6.4.E3双卡测试.sh) |

下面用 MNIST、种子 1234 验证 E0 的两种训练和测试入口。结果保存到独立的示例目录。每组训练 20 轮。

```bash
R0=E0/results/reproduce_mnist

bash "sh/3.1.E0单卡训练.sh" --dataset mnist --seed 1234 --output-dir "$R0"
bash "sh/3.3.E0双卡训练.sh" --dataset mnist --seed 1234 --output-dir "$R0"

bash "sh/3.2.E0单卡测试.sh" "$R0/mnist/checkpoints/e0_baseline_k1_ws1_mM_b256_r0_epoch20.pt"
bash "sh/3.4.E0双卡测试.sh" "$R0/mnist/checkpoints/e0_baseline_k1_ws2_mM_b128_r0_epoch20.pt"
```

训练脚本省略配置筛选时运行该卡数下的完整计划。E0、E1 默认将两种卡数的结果写到各自的 `results/reproduce/`。E2、E3 将双卡写到 `results/reproduce/`，单卡写到其 `single_gpu/` 子目录。复跑可传入新的 `--output-dir`。E2、E3 的 `--resume` 会跳过已完成且核对通过的运行。完整计划、筛选参数与检查点路径见 [E0](E0/README.md) · [E1](E1/README.md) · [E2](E2/README.md) · [E3](E3/README.md)。

## 验证与测试

训练每轮评估固定验证集，完成 20 轮后评估官方测试集并保存末轮模型。单卡测试、双卡测试脚本分别加载对应训练所得的检查点，均用 `cuda:0` 评估完整测试集。修改过训练输出目录、数据集或种子时，向测试脚本传入实际检查点路径。

E0—E3 的终端逐轮显示训练与验证损失、验证准确率、全局样本吞吐、平均更新与同步耗时及同步占比。E1—E3 同时显示计算耗时，E1 增加同步/计算耗时比，E2、E3 增加每样本同步耗时。E1—E3 的逐轮性能也写入 `epochs.csv`。扩展比、加速比和并行效率在单双卡配对后计算。

回归测试含 CUDA 模型检查与双 GPU NCCL 检查。

```bash
uv run --locked pytest -q
```

## 结果核验与绘图

先完成对应的完整训练计划，再进行汇总。各入口默认读取对应的复现目录。

| 实验 | 运行后使用的入口 |
|---|---|
| E0 | [绘图](sh/3.5.E0绘图.sh) |
| E1 | [结果核验](sh/4.4.E1结果核验.sh) · [绘图](sh/4.5.E1绘图.sh) |
| E2 | [结果核验](sh/5.5.E2结果核验.sh) · [配对统计](sh/5.6.E2配对统计.sh) · [绘图](sh/5.7.E2绘图.sh) |
| E3 | [结果核验](sh/6.5.E3结果核验.sh) · [配对统计](sh/6.6.E3配对统计.sh) · [绘图](sh/6.7.E3绘图.sh) · [数值诊断](sh/6.8.E3数值诊断.sh) |

使用自定义结果目录时，E0 绘图和 E1 核验/绘图接收 `--results-dir`，E2 与 E3 接收结果根目录作为位置参数。E2 与 E3 需要先配对统计，随后可在 `figs/` 生成图 1—7。E3 的数值诊断可独立运行，输出目录默认为 `E3/results/reproduce/numerics/`。E0 的全量核验、各脚本的完整用法见对应实验说明。

绘图脚本输出 300 DPI PNG。正式归档保存在各实验 `results/` 的数据集目录，E2、E3 的匹配单卡归档位于各自的 `results/single_gpu/`。
