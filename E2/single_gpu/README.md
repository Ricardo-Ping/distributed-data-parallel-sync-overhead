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

依赖安装和GPU检查见项目的[依赖安装](../../README.md#依赖安装)，训练前先[准备数据](../../README.md#数据准备)。本目录的训练使用一张CUDA GPU；比较与绘图读取已完成的单、双GPU结果。

在项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run python -m E2.single_gpu.run --list
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run python -m E2.single_gpu.run --resume
uv run python -m E2.single_gpu.check_results
uv run python -m E2.single_gpu.compare
uv run python -m E2.single_gpu.plot_results
```

可用`--dataset`、`--seed`和`--level`筛选，level指双卡每卡batch。重跑应指定新的`--output-dir`，已有同名目录不会覆盖。模型独立测试沿用`E2.evaluate`，检查点路径示例为`E2/results/single_gpu/mnist/ref128/seed1234/checkpoint.pt`。

独立结果目录通过核验的`--results-dir`指定单卡来源、`--dual-results-dir`指定对应双卡来源。比较入口使用同样两个参数，并以`--output-dir`选择统计保存目录；绘图的`--results-dir`指向该统计目录。默认均为当前正式结果。

结果在`../results/single_gpu/{数据集}/ref{参考batch}/seed{种子}/`；`steps.csv`保存每次完整更新，跨卡同步计数和字节为0。`epochs.csv`为20轮训练/验证，`final.json`为末轮指标，`checkpoint.pt`保存模型。环境和源码快照位于单卡结果根目录。

主结果目录的`scaling.csv`与`scaling_summary.csv`保存同种子配对的吞吐加速比、计算段加速比、并行效率、时间节省和质量差异。图5—7与图1—4共同编入[实验报告](../results/实验报告.md)的“实测结果”。
