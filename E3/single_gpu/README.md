# E3单GPU基线

单GPU保持全局batch256与对应双GPU相同的累积次数，实际micro-batch为双卡每卡值的两倍。

| 双GPU每卡micro | 单GPU micro | 累积K | 全局batch |
|---:|---:|---:|---:|
| 128 | 256 | 1 | 256 |
| 64 | 128 | 2 | 256 |
| 32 | 64 | 4 | 256 |
| 16 | 32 | 8 | 256 |

三数据集、三种子、20轮，共36次独立单卡训练。模型、优化器和数据设置沿用本实验`../configs/`；实际单卡Case及参考配置均保存于每次运行的`config.json`。全局样本顺序、每次更新的样本窗口、尾批和SGD次数与对应双卡一致。每个micro的loss按实际样本数/窗口样本数加权，整个窗口只更新一次。

依赖安装和GPU检查见项目的[依赖安装](../../README.md#依赖安装)，训练前先[准备数据](../../README.md#数据准备)。本目录的训练使用一张CUDA GPU；比较与绘图读取已完成的单、双GPU结果。

在项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run python -m E3.single_gpu.run --list
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run python -m E3.single_gpu.run --resume
uv run pytest E3/single_gpu/tests -q
uv run python -m E3.single_gpu.check_results
uv run python -m E3.single_gpu.compare
uv run python -m E3.single_gpu.plot_results
uv run python -m E3.evaluate --checkpoint E3/results/single_gpu/mnist/ref64/seed1234/checkpoint.pt --split test --device cuda:0
```

`--dataset`、`--seed`、`--level`筛选运行，level指双卡每卡micro；重跑指定独立`--output-dir`。训练包含每轮验证和末轮测试，核验会在CUDA上独立复测全部末轮模型。

重跑两组结果后，核验和比较均用`--results-dir`选择单卡目录、`--dual-results-dir`选择双卡目录；比较另有`--output-dir`保存CSV，绘图用`--results-dir`读取这些CSV。实际配置、样本窗口与测量标记仍须一致，不能只按文件名配对。

结果位于`../results/single_gpu/{数据集}/ref{参考micro}/seed{种子}/`。`steps.csv`以一次SGD为一行，记录窗口耗时、实际样本、micro数和零次跨卡同步；20轮验证、末轮测试、模型、环境与来源快照分别保存。单卡吞吐与双卡使用相同测量窗口比较，同种子配对后才计算三次结果的均值和样本标准差。

加速比、并行效率和图5—7统一编入[实验报告](../results/实验报告.md)的“实测结果”，与图1—4连续编号。本目录不导入其他实验的训练代码。
