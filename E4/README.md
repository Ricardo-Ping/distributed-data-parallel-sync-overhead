# E4：梯度累积中的通信优化

固定双GPU/NCCL、E0的MLP-M、每卡Micro-Batch=128，扫描K=1、2、4、8、16。每个K配对比较：`every_micro`每个micro反向都同步，`window_end`仅窗口最后一次反向同步；两者都在整个累积窗口结束后执行一次SGD。

三个数据集、三个种子、每次20轮，双GPU90次、匹配单GPU45次，共135次训练。只共享顶层[Data](../Data/README.md)原始读取接口，代码、划分、配置、模型、统计和结果均独立。实验设计与结果见[报告](results/实验报告.md)。

匹配的单GPU实现位于[single_gpu](single_gpu/README.md)：单卡micro256、同K、同全局batch256K与样本窗口，共45次训练。每份基线与同K两种双卡实现分别配对，得到90对结果。单卡结果在`results/single_gpu/`，加速比、计算段加速比和效率在`results/scaling.csv`；图1—8及配对统计统一位于报告“实测结果”。

## 运行环境

双GPU配置使用两张CUDA GPU和NCCL，匹配的单GPU基线使用一张卡。软硬件版本见项目的[运行环境](../README.md#运行环境)，依赖安装与GPU检查见[依赖安装](../README.md#依赖安装)。命令均在项目根目录执行，训练前按[数据说明](../Data/README.md)准备缓存。完整单、双GPU复现流程见[项目运行方法](../README.md#训练)。

## 运行

在项目根目录执行：

```bash
uv sync --locked
uv run python -m E4.run --list
CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=WARN uv run python -m E4.run

# 完整结果跳过；未完成目录先归档
uv run python -m E4.run --resume

# 单组合调试或整组重跑需使用新目录
uv run python -m E4.run --dataset mnist --level 4 --variant window_end --seed 1234 --output-dir E4/results/debug_run
uv run python -m E4.run --output-dir E4/results/new_run

uv run python -m E4.evaluate --checkpoint E4/results/mnist/k4_window_end/seed1234/checkpoint.pt --split validation --device cuda:0
uv run python -m E4.evaluate --checkpoint E4/results/mnist/k4_window_end/seed1234/checkpoint.pt --split test --device cuda:1
uv run pytest E4/tests -q
uv run python -m E4.check_results
uv run python -m E4.plot_results
```

核验与绘图支持`--results-dir`；核验默认在GPU上独立复测全部90个末轮模型，`--records-only`复算记录并检查GPU模型元数据。双卡训练需要两张可见GPU和NCCL。已有同名运行默认拒绝覆盖；训练前的端口占用最多重试三次，开始训练后不自动重试。

## 训练与指标

固定FP32、禁用TF32/AMP；SGD momentum=0.9，学习率MNIST/FashionMNIST为0.1、CIFAR-10为0.05，不随K缩放。分层留出10%验证、划分种子1234，训练种子1234/1235/1236。每轮完整使用训练样本并验证一次，20轮后测试一次并保存末轮模型。

尾批loss按micro实际样本数/本rank窗口实际样本数加权。`no_sync()`同时包住前K−1次forward与backward。每窗口只清空一次梯度、更新一次参数。K增加时全局batch=256K，更新次数减少；只有同K的同步方式对照匹配优化预算。

前16个micro用于预热，参与训练、排除性能统计；不完整窗口也训练但不计固定batch性能。记录数据、计算、同步、SGD与其他控制开销，显式等待CUDA完成，使用阻塞NCCL。一次窗口选择整体耗时较长的rank及其整行分段。逻辑载荷为实际归约buffer字节数累计，不是实际PCIe/网络链路流量。

## 结果结构

每次运行位于`results/{数据集}/k{K}_{同步方式}/seed{种子}/`，保存配置、划分、两个rank的运行与样本顺序哈希、逐窗口`steps_rank0/1.csv`、20轮`epochs.csv`、末轮模型、训练显存与`final.json`。

`config.py/configs`定义协议，`data.py`划分和采样，`model.py`模型保存加载，`distributed.py`CUDA/NCCL及同步hook，`train.py`训练验证测试，`evaluate.py`独立评估，`metrics.py`统计，`check_results.py`原始核验，`plot_results.py`仅生成图和CSV。

结果顶层包含环境、源码快照、训练日志、GPU复测、统计表与图表。图1—4分别展示同步次数/载荷、同步成本、吞吐/更新时间、准确率/显存；灰色为匹配单GPU，蓝色为双GPU每micro同步，橙色为双GPU窗口末同步，均保留三个种子与均值。同K两种双卡方式共用一条单卡基线，单卡跨卡通信指标为0。图5仅比较两种双卡同步方式的配对比值。

`main_comparison.csv`保存135次运行的主图数据，`main_comparison_sources.json`核对工作量与来源，`figure_validation.json`核对曲线和坐标范围。绘图默认读取所选目录的`single_gpu/`基线，可用`--single-results-dir`指定。所有图均不画置信区间。
