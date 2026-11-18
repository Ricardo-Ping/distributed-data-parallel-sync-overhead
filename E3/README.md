# E3：固定全局Batch Size，改变实现方式

E3对应实验B。固定两张GPU和全局batch=256，比较下列四种实现。模型、数据划分、学习率、样本顺序、参数更新次数和20轮训练预算一致。三个数据集、三个种子，双GPU与匹配单GPU各36次，共72次训练。

| GPU数 | 每卡Micro-Batch | 累积次数K | 每卡每次更新样本 | 全局Batch |
|---:|---:|---:|---:|---:|
| 2 | 128 | 1 | 128 | 256 |
| 2 | 64 | 2 | 128 | 256 |
| 2 | 32 | 4 | 128 | 256 |
| 2 | 16 | 8 | 128 | 256 |

代码、配置、测试和结果均独立位于`E3/`，只共享顶层[Data](../Data/README.md)原始数据接口。实验设计与结果见[报告](results/实验报告.md)。

匹配的单GPU实现位于[single_gpu](single_gpu/README.md)：micro256×1、128×2、64×4、32×8，保持同全局batch256、累积次数与样本窗口，共36次单卡训练，结果在`results/single_gpu/`。同种子加速比、计算段加速比和并行效率保存于`results/scaling.csv`；图1—7及配对统计统一位于报告“实测结果”。

## 运行环境

双GPU配置使用两张CUDA GPU和NCCL，匹配的单GPU基线使用一张卡。软硬件版本见项目的[运行环境](../README.md#运行环境)，依赖安装与GPU检查见[依赖安装](../README.md#依赖安装)。命令均在项目根目录执行，训练前按[数据说明](../Data/README.md)准备缓存。完整单、双GPU复现流程见[项目运行方法](../README.md#训练)。

## 累积的正确实现

采样器先为每个rank构造相同的128样本更新窗口，再在主机侧按micro-batch分片，逐片传输至GPU。这样四种实现的全局样本组与更新边界保持一致。

每个窗口仅清空梯度一次、执行一次SGD更新；前K−1个micro-batch的forward和backward均放入`no_sync()`，最后一个micro-batch才同步累计梯度。每个micro loss乘以“该micro实际样本数/该rank窗口实际样本数”。完整窗口等价于除以K；尾批不直接除以固定K。

每次参数更新都恰好发生一次完整梯度AllReduce。因此这组实验里，改变K不会减少每轮通信调用总数，也不会改变每次更新的梯度载荷，主要比较micro-batch计算效率、额外执行成本、显存与吞吐。

所有模型计算使用CUDA，多卡通信统一NCCL。沿用E0的MLP-M、数据划分、三个种子、学习率、FP32和20轮末轮测试协议。保留尾批完成训练；性能统计排除前10次参数更新和各轮尾批。阻塞式同步计时不代表充分重叠计算通信的最优DDP吞吐。

## 运行

在项目根目录执行：

```bash
uv sync --locked
uv run python -m E3.run --list
CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=WARN uv run python -m E3.run
uv run python -m E3.run --resume

# 重跑或调试使用独立输出目录
uv run python -m E3.run --output-dir E3/results/new_run
uv run python -m E3.run --dataset mnist --batch 16 --seed 1234 --output-dir E3/results/debug_run

# 验证或测试已有模型
uv run python -m E3.evaluate --checkpoint E3/results/mnist/b16_k8/seed1234/checkpoint.pt --split validation --device cuda:0
uv run python -m E3.evaluate --checkpoint E3/results/mnist/b16_k8/seed1234/checkpoint.pt --split test --device cuda:1

uv run pytest E3/tests -q
uv run python -m E3.check_results
uv run python -m E3.compare_implementations
uv run python -m E3.verify_numerics
uv run python -m E3.plot_results
```

核验、参数差异比较和绘图支持`--results-dir`。完整核验会在GPU上独立复测36个模型；`check_results --records-only`只复算记录并检查模型元数据。`compare_implementations`输出20轮后各实现相对128×1的参数差异及测试指标差异；它记录浮点差异，不要求长程训练逐位相同。逐次梯度与含momentum的参数更新等价性由GPU前置测试检查。

`--resume`跳过身份、配置及末轮协议一致的完整结果，拒绝覆盖未完成目录；未完成运行应先归档。初始化阶段的端口占用最多尝试三次，训练开始后不自动重试。源码快照检查禁止在训练、计时或同步逻辑变化后混合续跑。

`verify_numerics`用真实完整窗口和尾批，在相同参数、momentum状态下比较正式FP32路径与整批参考，并额外使用GPU FP64核对梯度归一化；输出`numerical_equivalence.csv`。该诊断不计入36次正式训练，也不参与性能统计。FP32逐元素差异、ReLU分支差异与末轮参数差异的说明见报告。

## 结果位置

每次运行位于`results/{数据集}/b{micro}_k{累积}/seed{种子}/`，单独保存配置、数据划分、两个rank的运行元数据、样本顺序哈希、逐更新CSV、20轮训练验证结果、显存、最终测试和末轮检查点。

`final.csv`包含36次完整结果；`seed_summary.csv`包含三种子均值、样本标准差与原始范围；`paired_relative.csv`逐种子比较相对128×1的吞吐及更新时间；`implementation_differences.csv`记录模型参数和测试指标差异。源码快照与环境位于`provenance/`和`environment.json`。

图1—4直接比较单、双GPU的吞吐、更新与计算段成本、跨卡同步成本、显存和末轮测试结果。灰色为单GPU、蓝色为双GPU，形状区分三个种子，粗线/菱形为均值。横轴标注双GPU的每卡micro×K，单GPU实际micro为其两倍，两组全局Batch均为256且K相同。单GPU跨卡同步指标为0，不绘制置信区间。

`main_comparison.csv`保存72次运行的主图数据，`main_comparison_sources.json`核对配置、工作量与文件来源，`figure_validation.json`核对种子、均值和坐标范围。基线默认读取所选结果目录的`single_gpu/`，可用`--single-results-dir`指定其他位置。
