# E3：固定全局Batch Size，改变实现方式

E3对应实验B。固定两张GPU和全局batch=256，比较下列四种实现。模型、数据划分、学习率、样本顺序、参数更新次数和20轮训练预算一致。三个数据集、三个种子，双GPU与匹配单GPU各36次，共72次训练。

| GPU数 | 每卡Micro-Batch | 累积次数K | 每卡每次更新样本 | 全局Batch |
|---:|---:|---:|---:|---:|
| 2 | 128 | 1 | 128 | 256 |
| 2 | 64 | 2 | 128 | 256 |
| 2 | 32 | 4 | 128 | 256 |
| 2 | 16 | 8 | 128 | 256 |

代码、配置、测试和结果均独立位于`E3/`，通过顶层 [Data](../Data/README.md) 读取原始数据。实验设计与结果见[报告](results/实验报告.md)。

匹配的单GPU实现位于[single_gpu](single_gpu/README.md)：micro256×1、128×2、64×4、32×8，保持同全局batch256、累积次数与样本窗口，共36次单卡训练，结果在`results/single_gpu/`。同种子加速比、计算段加速比和并行效率保存于`results/scaling.csv`；图1—7及配对统计统一位于报告“实测结果”。

## 运行

命令在项目根目录执行。两种卡数各有 36 次完整训练。先用 `--list` 查看计划，训练命令依次执行。

```bash
bash "sh/6.1.E3单卡训练.sh" --list
bash "sh/6.3.E3双卡训练.sh" --list

bash "sh/6.3.E3双卡训练.sh"
bash "sh/6.1.E3单卡训练.sh"
```

双卡默认写入 `E3/results/reproduce/`，单卡默认写入该目录的 `single_gpu/`。自定义输出时也保持这一目录关系，便于核验和配对。增加 `--resume` 可跳过配置、身份和末轮协议一致的完整运行。未完成的运行可归档后重跑，或使用新的 `--output-dir`。双卡启动进程组时遇端口占用最多尝试三次，训练开始后遇异常即停止。

下面复现 MNIST、种子 1234 的代表配置，写入独立目录。双卡的 `--batch 32` 设置每卡 micro-batch，单卡的 `--level 32` 选择匹配双卡档位。单卡实际 micro-batch 为 64，两组全局 batch 都是 256，累积 K=4。

```bash
R=E3/results/reproduce_mnist

bash "sh/6.3.E3双卡训练.sh" --dataset mnist --batch 32 --seed 1234 --output-dir "$R"
bash "sh/6.4.E3双卡测试.sh" "$R/mnist/b32_k4/seed1234/checkpoint.pt"

bash "sh/6.1.E3单卡训练.sh" --dataset mnist --level 32 --seed 1234 --output-dir "$R/single_gpu"
bash "sh/6.2.E3单卡测试.sh" "$R/single_gpu/mnist/ref32/seed1234/checkpoint.pt"
```

两个测试脚本均在单张 GPU 上复测所选的末轮检查点。用命令中的训练结果目录构造路径，可以复测不同配置和种子。

完成 **36 对**完整训练后，按以下顺序复算、配对和绘图。位置参数选择包含单双卡数据的结果根目录。绘图读取配对 CSV，在 `figs/` 生成 7 张统计图。

```bash
R=E3/results/reproduce

uv run --locked pytest E3/tests E3/single_gpu/tests -q
bash "sh/6.5.E3结果核验.sh" "$R"
bash "sh/6.6.E3配对统计.sh" "$R"
bash "sh/6.7.E3绘图.sh" "$R"
```

核验脚本同时复算单双卡记录并在 CUDA 上复测全部末轮模型。只核对双卡原始记录时，可运行 `uv run --locked python -m E3.check_results --results-dir "$R" --records-only`。

E3 还生成 `implementation_differences.csv`，记录各双卡配置相对 128×1 的末轮参数和测试差异。

数值诊断独立于正式训练，可以单独执行。默认输出到 `E3/results/reproduce/numerics/`，也可传入其他目录作为位置参数。

```bash
bash "sh/6.8.E3数值诊断.sh"
```

诊断比较相同初值和 momentum 下的 FP32 梯度及参数更新，并以 CUDA FP64 的整批参考检查梯度归一化。终端汇总 24 条完整窗口和 24 条尾窗口的逐 rank 结果，详细记录写入 `numerical_equivalence.csv`。FP32 的舍入和 ReLU 分支差异详见实验报告。

## 结果位置

每次运行位于`results/{数据集}/b{micro}_k{累积}/seed{种子}/`，单独保存配置、数据划分、两个rank的运行元数据、样本顺序哈希、逐更新CSV、20 轮训练验证结果、显存、最终测试和末轮检查点。新运行的 `epochs.csv` 同时保存样本吞吐、更新/计算/同步耗时、同步占比和每样本同步耗时。

`final.csv`包含36次完整结果；`seed_summary.csv`包含三种子均值、样本标准差与原始范围；`paired_relative.csv`逐种子比较相对128×1的吞吐及更新时间；`implementation_differences.csv`记录模型参数和测试指标差异。源码快照与环境位于`provenance/`和`environment.json`。

图1—4直接比较单、双GPU的吞吐、更新与计算段成本、跨卡同步成本、显存和末轮测试结果。灰色为单GPU、蓝色为双GPU，形状区分三个种子，粗线/菱形为均值。横轴标注双GPU的每卡micro×K，单GPU实际micro为其两倍，两组全局Batch均为256且K相同。单GPU跨卡同步指标为0，不绘制置信区间。

`main_comparison.csv`保存72次运行的主图数据，`main_comparison_sources.json`核对配置、工作量与文件来源，`figure_validation.json`核对种子、均值和坐标范围。基线默认读取所选结果目录的`single_gpu/`，可用`--single-results-dir`指定其他位置。
