# E1：模型规模对计算与通信比例的影响

E1在独立目录中实现数据划分、训练、验证、末轮测试、计时、模型保存和结果核验，只通过顶层[Data](../Data/README.md)读取原始数据。

本实验沿用 E0 的两层隐藏层 MLP，固定每张 GPU 的 batch=128，比较 S、M、L 三种宽度。每个数据集内只改变模型宽度；每种模型都运行单卡、双卡和三个种子，共54次正式运行，每次20轮。实验设计与结果解读见 [实验报告](results/实验报告.md)。

## 运行环境

完整对照使用两张CUDA GPU，通信后端为NCCL；单GPU配置使用一张卡。软硬件版本见项目的[运行环境](../README.md#运行环境)，依赖安装与GPU检查见[依赖安装](../README.md#依赖安装)。以下命令均在项目根目录执行，训练前按[数据说明](../Data/README.md)准备三个数据集的缓存。

## 固定项与变化项

| 项目 | 设置 |
|---|---|
| 模型族 | Flatten → Linear → ReLU → Linear → ReLU → Linear；输出10类 Logits |
| 规模 | S：256→128；M：1024→512；L：4096→2048；宽度每档乘4，隐藏层数和宽度比例不变 |
| 数据集 | MNIST、FashionMNIST、CIFAR-10；分别分析，输入维数在各数据集内固定 |
| 数据划分 | 官方训练集分层留10%验证，划分种子1234；官方测试集独立保留 |
| 每卡 batch | 128；单卡全局128、双卡全局256；丢弃每轮不完整批次，确保计时批量完全一致 |
| 训练 | 固定20轮，SGD momentum=0.9；MNIST/FashionMNIST lr=0.1，CIFAR-10 lr=0.05 |
| 精度 | FP32，禁用 TF32、AMP 和梯度压缩，不使用梯度累积或编译 |
| 通信 | 一进程一张卡，NCCL；DDP 梯度桶容量固定128 MiB，所有模型均只需一个完整梯度桶 |
| 计时 | 阻塞同步 hook，CUDA 显式等待；每次运行前10次更新参与训练，但排除性能统计 |
| 其他 | 每进程4个主机计算线程，数据加载 worker=0；评估 batch=1024，种子1234/1235/1236 |

MLP 没有序列长度设置。固定的是图像输入维数、局部批量、层数、激活函数和训练协议。双卡只分担样本，每张卡仍保存完整模型。

E0固定全局batch=256，单卡每批256、双卡每卡128；E1固定每卡128。E1的M与E0网络一致，但单卡批量、尾批处理和性能统计协议不同。E1的单、双卡均独立训练，L的隐藏层宽度为4096→2048。

## 指标

记预热后每次更新选定的分段时间总和为 `T`，计算段总和为 `C`，同步段总和为 `S`，更新数为 `U`，全局样本数为 `N`。分段含义和局限见报告。

| 指标 | 计算方式 | 用途 |
|---|---|---|
| 平均计算耗时 | `C/U`，毫秒 | 比较前向和反向随模型规模的变化，已排除 hook 同步段 |
| 平均同步耗时 | `S/U`，毫秒 | 比较梯度同步路径成本，含等待、AllReduce、梯度平均和设备等待 |
| 同步占比 | `S/T` | 看通信路径占整个更新的比例 |
| 同步/计算时间比 | `S/C` | 直接对照同步与前向反向耗时，主研究指标之一 |
| 计算同步二者中的同步比例 | `S/(C+S)` | 排除数据准备和优化器操作，补充分段比较 |
| 平均及P50/P95更新耗时 | `T/U`及逐步分位数 | 看总体更新成本和长尾波动 |
| 分段训练吞吐 | `N/(T/1000)`，样本/秒 | 固定局部批量下的处理速度 |
| 双卡吞吐扩展比 | 同种子的 `Q双/Q单` | 每次更新工作量随卡数翻倍，理想值为2 |
| 双卡弱扩展效率 | `(Q双/Q单)/2` | 先逐种子配对计算，再取三个种子的均值 |
| 参数量、FP32梯度载荷 | `P`、`4P`字节 | 解释计算规模与每次梯度归约的逻辑数据量 |
| 每卡训练显存峰值 | 各卡 `max_memory_allocated` 的最大值，MB | 检查宽度增大带来的资源成本，不等同于全部显存占用 |
| 验证/最终测试损失、准确率 | 固定验证集/官方测试集 | 确认模型正常学习，辅助观察；不同全局batch的准确率不作优化等价比较 |

三个种子先各自计算完整运行指标，再等权取均值和样本标准差。逐步计时样本不是独立的训练重复；图中展示三种子原值与均值，不绘制置信区间。

## 运行

在项目根目录执行：

```bash
uv sync --locked
nvidia-smi

# 只列计划，不读入数据或启动训练
uv run python -m E1.run --list

# 正式54次运行；目录中已有对应记录时会拒绝覆盖
CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=WARN uv run python -m E1.run

# 重跑使用新目录
CUDA_VISIBLE_DEVICES=0,1 uv run python -m E1.run --output-dir E1/results/new_run

# 仅选择某个数据集、规模、种子或卡数
uv run python -m E1.run --dataset mnist --model S --seed 1234 --world-size 2 --output-dir E1/results/debug_run
```

完整计划按种子轮换模型次序，同时交替单卡、双卡先后，所有运行串行执行。上述单组合筛选只用于调试，不代替完整54次结果。

```bash
# 模型计算检查使用 CUDA，双卡检查使用 NCCL
uv run pytest E1/tests -q

# 重算全部记录并在GPU上独立测试54个模型
uv run python -m E1.check_results

# 只复算记录、核验GPU运行元数据与模型元数据
uv run python -m E1.check_results --records-only

# 用现有末轮模型独立评估验证集或测试集
uv run python -m E1.evaluate --checkpoint E1/results/mnist/M/ws2_seed1234/checkpoint.pt --split validation --device cuda:0
uv run python -m E1.evaluate --checkpoint E1/results/mnist/M/ws2_seed1234/checkpoint.pt --split test --device cuda:1

# 生成图表与三种子统计表
uv run python -m E1.plot_results
```

核验和绘图均支持 `--results-dir E1/results/new_run`。绘图要求有三个数据集、三个模型、单/双卡、三个种子的完整结果，需要可用中文字体，例如 Noto Sans CJK。

## 文件

```text
E1/
├── configs/                         三个数据集的独立配置
├── data.py                          本实验分层划分与固定局部批量采样
├── model.py                         S/M/L架构与检查点
├── distributed.py                   CUDA/NCCL与阻塞式梯度同步
├── train.py、evaluate.py             训练、验证、最终测试和独立评估
├── metrics.py                       原始记录与指标复算
├── run.py                           配对计划、启动与代码快照
├── check_results.py、plot_results.py 核验、图表和统计表
├── tests/                           数据协议、指标与GPU正确性检查
└── results/
    ├── mnist/S/ws1_seed1234/         每个组合独立目录，另有其他数据集/模型/卡数/种子
    │   ├── config.json、dataset.json、split_indices.npz
    │   ├── runtime_rank0.json        双卡另有rank1
    │   ├── steps_rank0.csv           每张卡的逐步时间、实际批量、梯度载荷
    │   ├── memory_rank0.json         训练分段内显存峰值
    │   ├── epochs.csv               每轮训练/验证，不能包含逐轮测试
    │   ├── final.json               最终测试与预热后性能
    │   └── checkpoint.pt            第20轮模型
    ├── provenance/                  实际运行源码快照与SHA-256
    ├── environment.json             设备、版本、拓扑与协议
    ├── final.csv、seed_summary.csv   54次结果与三种子统计
    ├── paired_scaling.csv           每个种子的单/双卡配对扩展指标
    ├── model_sizes.csv、figs/        模型大小与实验图
    └── 实验报告.md                  实验设计、指标、结果与局限
```
