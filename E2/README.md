# E2：固定GPU数量，改变每卡Batch Size

E2对应实验A。固定两张RTX 4090、MLP-M和梯度累积次数1，扫描每卡batch=32、64、128、256、512，对应全局batch=64、128、256、512、1024。三个数据集、三个种子、每次20轮，双GPU与匹配单GPU各45次，共90次训练。

代码、配置、测试与结果独立保存在本目录，只调用顶层[Data](../Data/README.md)读取原始数据。实验设计与结果见[报告](results/实验报告.md)。

## 运行环境

双GPU配置使用两张CUDA GPU和NCCL，匹配的单GPU基线使用一张卡。软硬件版本见项目的[运行环境](../README.md#运行环境)，依赖安装与GPU检查见[依赖安装](../README.md#依赖安装)。命令均在项目根目录执行，训练前按[数据说明](../Data/README.md)准备缓存。完整单、双GPU复现流程见[项目运行方法](../README.md#训练)。

## 固定与变化的设置

每卡batch是本实验唯一主动扫描的训练参数。固定两张GPU、FP32、1024→512的两层隐藏层、ReLU、SGD momentum=0.9和学习率。MNIST、FashionMNIST学习率0.1，CIFAR-10为0.05，不按batch缩放。沿用E0的分层验证划分，划分种子1234，训练种子1234、1235、1236。

训练保留尾批，完整使用每轮训练样本；每轮验证，20轮结束后测试一次并保存末轮模型。性能统计排除前10次参数更新及各轮尾批，不排除这些样本的训练。双卡每次参数更新只进行一次完整梯度AllReduce，NCCL阻塞同步；通信桶固定25 MiB。

计算段效率使用“每卡计算段吞吐（样本/计算秒）”，不能当作SM利用率或峰值算力利用率。整体吞吐、每次更新耗时、同步耗时/占比、每样本同步成本、显存及最终分类结果的定义见报告。

## 运行入口

在项目根目录执行：

```bash
uv sync --locked
uv run python -m E2.run --list

# 正式45次训练，已有同名运行时会拒绝覆盖
CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=WARN uv run python -m E2.run

# 中断后跳过已完整保存的运行；未完成目录须先归档
uv run python -m E2.run --resume

# 重跑另选结果目录
uv run python -m E2.run --output-dir E2/results/new_run

# 单个组合的调试
uv run python -m E2.run --dataset mnist --batch 128 --seed 1234 --output-dir E2/results/debug_run

# 独立验证或测试，模型计算均使用CUDA
uv run python -m E2.evaluate --checkpoint E2/results/mnist/b128_k1/seed1234/checkpoint.pt --split validation --device cuda:0
uv run python -m E2.evaluate --checkpoint E2/results/mnist/b128_k1/seed1234/checkpoint.pt --split test --device cuda:1

uv run pytest E2/tests -q
uv run python -m E2.check_results
uv run python -m E2.plot_results
```

`check_results`和`plot_results`支持`--results-dir`。核验默认复算记录并独立GPU复测全部模型；`--records-only`只复算原始记录与检查模型元数据。所有模型计算必须使用GPU，缺少CUDA或NCCL时明确报错。

启动进程组时若端口被占用，且目录仍只有配置文件，会重新选择端口，最多尝试三次。已开始训练的运行不会自动重试。`--resume`核对已有完整结果的配置、身份和末轮协议后跳过；训练、计时或同步源码变化时拒绝续跑。

## 文件与图表

`config.py`与`configs/`定义协议，`data.py`负责划分和全局更新窗口采样，`model.py`定义MLP-M，`distributed.py`实现CUDA/NCCL及同步计时，`train.py`完成训练验证和末轮测试，`evaluate.py`支持独立评估，`metrics.py`从逐rank记录计算性能。训练流程只调用E2和Data内的模块。

每个运行位于`results/{数据集}/b{每卡batch}_k1/seed{种子}/`，包含：

- `config.json`、`dataset.json`、`split_indices.npz`：实际配置与数据划分。
- `runtime_rank0/1.json`、`sampling_rank0/1.json`：设备、后端及各轮样本顺序哈希。
- `steps_rank0/1.csv`：按参数更新记录的分段时间、样本数、同步调用与载荷。
- `epochs.csv`：20轮训练和验证结果；不包含逐轮测试。
- `memory_rank0/1.json`、`final.json`、`checkpoint.pt`：训练显存峰值、末轮测试及模型。

顶层`results/`保存环境、源码快照、训练日志、核验记录和图表。`final.csv`为45次结果，`seed_summary.csv`保留三种子均值、样本标准差与观测范围，`paired_relative.csv`为相同种子相对每卡128基线的吞吐和更新时间比。

图1—4均直接比较相同全局Batch的单、双GPU：依次展示吞吐，更新耗时与计算段吞吐，跨卡同步成本，显存与末轮准确率。灰色为单GPU、蓝色为双GPU；形状区分三个种子，同色粗线/菱形为均值，单卡均值用虚线。单GPU的跨卡同步指标为0，不画置信区间。

`main_comparison.csv`保存90次运行的主图数据，`main_comparison_sources.json`记录配对配置和原始文件哈希，`figure_validation.json`核对每条种子/均值曲线与坐标范围。绘图默认从所选结果目录的`single_gpu/`读取基线，也可用`--single-results-dir`指定。

## 匹配单GPU基线

单GPU设置45次独立训练：实际batch为64、128、256、512、1024，与双GPU对应配置保持相同全局batch、样本窗口、SGD次数及20轮。实现位于`single_gpu/`，结果位于`results/single_gpu/`。每次运行的`config.json`同时保存实际单卡配置与双卡参考配置。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run python -m E2.single_gpu.run --list
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run python -m E2.single_gpu.run --resume
uv run python -m E2.single_gpu.check_results
uv run python -m E2.single_gpu.compare
uv run python -m E2.single_gpu.plot_results
```

`scaling.csv`保留45对同种子结果，`scaling_summary.csv`汇总吞吐加速比、计算段加速比、并行效率和时间节省比例。图5—7依次展示吞吐/更新时间、并行收益、质量/显存，与图1—4共同编入报告“实测结果”。逐步吞吐和同步记录位于`results/step_metrics/`。
