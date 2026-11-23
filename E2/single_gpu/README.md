# E2单GPU基线

本目录实现与E2双GPU五档配置逐一匹配的单GPU基线。使用一个CUDA设备、一个训练进程和一个MLP-M，不建立分布式进程组。

| 双GPU每卡batch | 单GPU实际batch | 全局batch | 累积次数 |
|---:|---:|---:|---:|
| 32 | 64 | 64 | 1 |
| 64 | 128 | 128 | 1 |
| 128 | 256 | 256 | 1 |
| 256 | 512 | 512 | 1 |
| 512 | 1024 | 1024 | 1 |

三数据集、三种子、20轮，共45次独立训练。基础设置读取本实验`../configs/`，实际单卡配置由`config.py`中的Case定义，并与参考配置一起写入每次运行的`config.json`。划分、全局样本顺序、更新边界和尾批与对应双卡保持一致。每轮验证，末轮测试一次并保存模型。

环境和数据按项目说明准备。此训练入口绑定 GPU 0，直接读取本实验配置。操作步骤见 [项目 README](../../README.md#依赖安装)。

在项目根目录执行。下面列出单卡计划，并复现 MNIST、种子 1234 的代表配置。

```bash
R=E2/results/reproduce_mnist

bash "sh/5.1.E2单卡训练.sh" --list
bash "sh/5.1.E2单卡训练.sh" --dataset mnist --level 128 --seed 1234 --output-dir "$R/single_gpu"
bash "sh/5.2.E2单卡测试.sh" "$R/single_gpu/mnist/ref128/seed1234/checkpoint.pt"
```

这里 `--level 128` 对应双卡每卡 batch 128，单卡实际 batch 256，K=1。 省略 `--dataset`、`--level` 和 `--seed` 运行完整单卡计划，默认写入 `E2/results/reproduce/single_gpu/`。使用 `--resume` 可跳过检查通过的完整运行，重跑时可传入新的 `--output-dir`。测试脚本加载第 20 轮模型，在 `cuda:0` 上复测官方测试集。

完整的核验、配对和绘图需要同时准备匹配的双卡结果。先把单卡结果放在双卡结果根目录的 `single_gpu/`，再按照主说明运行后处理脚本。操作步骤见 [单双卡流程](../README.md#运行入口)。

结果在`../results/single_gpu/{数据集}/ref{参考batch}/seed{种子}/`。`steps.csv` 保存每次参数更新，包括尾批更新。跨卡同步次数与载荷均为 0。`epochs.csv` 保存 20 轮训练/验证指标，新运行还写入每轮性能，同步相关字段为 0。`final.json`为末轮指标，`checkpoint.pt`保存模型。环境和源码快照位于单卡结果根目录。

主结果目录的`scaling.csv`与`scaling_summary.csv`保存同种子配对的吞吐加速比、计算段加速比、并行效率、时间节省和质量差异。图5—7与图1—4共同编入[实验报告](../results/实验报告.md)的“实测结果”。
