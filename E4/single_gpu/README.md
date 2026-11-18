# E4单GPU基线

双GPU每卡micro固定128，单GPU实际micro固定256；两者按相同K=1、2、4、8、16累积，全局batch均为256K。全局样本顺序、更新窗口、实际尾批与20轮预算匹配。单GPU每个窗口只执行一次SGD，跨卡归约为0。

同K的every_micro与window_end共用一份真实单卡基线，因为它们只在跨卡同步时机上不同；单卡不存在这种区别。三数据集、五档K、三种子共45次单卡训练，对应90对单/双卡比较，不能把复用基线算成90次单卡训练。

实际配置由`config.py`的Case和本实验`../configs/`共同确定，每次运行完整保存在`config.json`。模型、学习率与momentum不变，按实际micro样本数加权loss，保留尾批；每轮验证，20轮后测试一次并保存末轮模型。性能沿用前16个micro预热和完整窗口筛选规则。

依赖安装和GPU检查见项目的[依赖安装](../../README.md#依赖安装)，训练前先[准备数据](../../README.md#数据准备)。本目录的训练使用一张CUDA GPU；比较与绘图读取已完成的单、双GPU结果。

在项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run python -m E4.single_gpu.run --list
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run python -m E4.single_gpu.run --resume
uv run pytest E4/single_gpu/tests -q
uv run python -m E4.single_gpu.check_results
uv run python -m E4.single_gpu.compare
uv run python -m E4.single_gpu.plot_results
uv run python -m E4.evaluate --checkpoint E4/results/single_gpu/mnist/ref4/seed1234/checkpoint.pt --split test --device cuda:0
```

可用`--dataset`、`--seed`和`--level`筛选，level为K。`--resume`跳过完整运行，未完成目录需先归档；重跑用新的`--output-dir`。结果在`../results/single_gpu/{数据集}/ref{K}/seed{种子}/`，每次更新及全部micro的成本记录于`steps.csv`，其他文件包括20轮验证、末轮测试、模型、采样哈希和运行元数据。

核验和比较支持`--results-dir`选择单卡来源、`--dual-results-dir`选择双卡来源；比较的`--output-dir`指定统计目录，绘图的`--results-dir`读取该目录。两个同步方式仍从同一批双卡目录取数，并与同一份单卡记录分别匹配。

`scaling.csv`保存与两种双卡实现分别配对的吞吐加速比、计算段加速比、效率与质量差异。图6—8及波动解释统一编入[实验报告](../results/实验报告.md)的“实测结果”，与图1—5连续编号。
