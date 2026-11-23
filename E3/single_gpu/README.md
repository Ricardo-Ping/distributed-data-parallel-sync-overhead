# E3单GPU基线

单GPU保持全局batch256与对应双GPU相同的累积次数，实际micro-batch为双卡每卡值的两倍。

| 双GPU每卡micro | 单GPU micro | 累积K | 全局batch |
|---:|---:|---:|---:|
| 128 | 256 | 1 | 256 |
| 64 | 128 | 2 | 256 |
| 32 | 64 | 4 | 256 |
| 16 | 32 | 8 | 256 |

三数据集、三种子、20轮，共36次独立单卡训练。模型、优化器和数据设置沿用本实验`../configs/`；实际单卡Case及参考配置均保存于每次运行的`config.json`。全局样本顺序、每次更新的样本窗口、尾批和SGD次数与对应双卡一致。每个micro的loss按实际样本数/窗口样本数加权，整个窗口只更新一次。

环境和数据按项目说明准备。此训练入口绑定 GPU 0，直接读取本实验配置。操作步骤见 [项目 README](../../README.md#依赖安装)。

在项目根目录执行。下面列出单卡计划，并复现 MNIST、种子 1234 的代表配置。

```bash
R=E3/results/reproduce_mnist

bash "sh/6.1.E3单卡训练.sh" --list
bash "sh/6.1.E3单卡训练.sh" --dataset mnist --level 32 --seed 1234 --output-dir "$R/single_gpu"
bash "sh/6.2.E3单卡测试.sh" "$R/single_gpu/mnist/ref32/seed1234/checkpoint.pt"
```

这里 `--level 32` 对应双卡每卡 32×4，单卡实际使用 64×4，两者全局 batch 均为 256。 省略 `--dataset`、`--level` 和 `--seed` 运行完整单卡计划，默认写入 `E3/results/reproduce/single_gpu/`。使用 `--resume` 可跳过检查通过的完整运行，重跑时可传入新的 `--output-dir`。测试脚本加载第 20 轮模型，在 `cuda:0` 上复测官方测试集。

完整的核验、配对和绘图需要同时准备匹配的双卡结果。先把单卡结果放在双卡结果根目录的 `single_gpu/`，再按照主说明运行后处理脚本。操作步骤见 [单双卡流程](../README.md#运行)。

结果位于`../results/single_gpu/{数据集}/ref{参考micro}/seed{种子}/`。`steps.csv`以一次SGD为一行，记录窗口耗时、实际样本、micro数和零次跨卡同步；`epochs.csv` 保存 20 轮训练/验证指标，新运行还写入每轮性能，同步相关字段为 0。末轮测试、模型、环境与源码快照分别保存。单卡吞吐与双卡使用相同测量窗口比较，同种子配对后才计算三次结果的均值和样本标准差。

加速比、并行效率和图5—7统一编入[实验报告](../results/实验报告.md)的“实测结果”，与图1—4连续编号。本目录不导入其他实验的训练代码。
