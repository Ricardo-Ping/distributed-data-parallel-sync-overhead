# E5：Local SGD的同步间隔

固定双GPU/NCCL、E0的MLP-M、每卡Batch=128，每个本地batch立即执行一次SGD。扫描H=1、2、4、8、16，表示两次参数平均之间允许的本地更新次数。H=1为逐步参数平均基线；各rank的momentum始终保持本地，不同步梯度或momentum。每轮末不足H步也进行参数平均，再验证共同模型。

三个数据集、三个种子、每次20轮，双卡主实验45次，单卡参考54次，共99次完整训练与末轮模型CUDA独立复测。代码、配置、测试和结果独立，仅共享顶层[Data](../Data/README.md)原始读取接口。实验设计与结果见[报告](results/实验报告.md)。

每个本地SGD步记录于`steps_rank*.csv`，H步同步轮次记录于`rounds_rank*.csv`。[single_gpu](single_gpu/README.md)分别实现45次串行双副本Local SGD基线、9次普通单模型SGD任务参考；后者H>1时算法不同，分别报告。配对启动器交错安排单、双卡任务，正式采集依次串行；命令与结果位置见单卡说明。

## 运行环境

完整配对实验需要两张CUDA GPU，跨卡参数平均使用NCCL；两类单GPU参考各使用一张卡。软硬件版本见项目的[运行环境](../README.md#运行环境)，依赖安装与GPU检查见[依赖安装](../README.md#依赖安装)。以下命令均在项目根目录执行，训练前按[数据说明](../Data/README.md)准备缓存。完整的配对运行命令见[项目运行方法](../README.md#训练)。

## 运行

项目根目录执行：

```bash
uv sync --locked
uv run python -m E5.run --list
CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=WARN uv run python -m E5.run
uv run python -m E5.run --resume
uv run python -m E5.run --dataset mnist --level 8 --seed 1234 --output-dir E5/results/debug_run
uv run python -m E5.run --output-dir E5/results/new_run

uv run python -m E5.evaluate --checkpoint E5/results/mnist/h8/seed1234/checkpoint.pt --split validation --device cuda:0
uv run python -m E5.evaluate --checkpoint E5/results/mnist/h8/seed1234/checkpoint.pt --split test --device cuda:1
uv run pytest E5/tests -q
uv run python -m E5.check_results

# 性能采集全部结束后，单独采集GPU活动时间线，不能与正式训练并行
uv run python -m E5.profile_activity
uv run python -m E5.plot_results
```

核验、Profiler、绘图支持`--results-dir`。核验默认在GPU上复测全部45个末轮模型。训练前端口冲突最多尝试三次，已开始训练的运行不自动重试。已有结果不覆盖；`--resume`仅跳过配置和身份一致的完整运行，未完成目录先归档。

## 控制变量与计时

固定FP32、SGD momentum=0.9、学习率MNIST/FashionMNIST为0.1、CIFAR-10为0.05，禁用TF32/AMP。沿用E0的验证划分与三个种子，每轮保留尾批完成全部训练，验证共同参数，20轮后才测试并保存末轮模型。

每次本地更新的跨卡样本总数固定256，H步的样本量256H不等于一次有效大batch：H>1时参数在本地已更新多次。通过NCCL平均参数，动量不平均；该选择固定于所有H。

以H步加一次参数平均构成计时轮次，选择整轮耗时较大的rank及其整行分段，避免把未同步的每个本地step分别取最大值再拼接。前16步预热、含尾批或不足H步的轮次参与训练但排除稳态性能。每epoch最后一个完整H步轮次，在平均前测一次参数漂移；诊断时间独立记录并从性能计时扣除。

Time-to-Target从训练循环开始计时，含训练、验证、固定频率漂移诊断、日志与轮次控制，不含数据预处理和进程组初始化。首次验证准确率达到MNIST97%、FashionMNIST85%、CIFAR-1045%时记录时间；20轮未达到保留未达状态，不填0或剔除。目标在正式运行前固定，不看测试集选阈值。

独立Profiler对每个配置按种子重新初始化模型，固定16步预热、64步采集，覆盖三个数据集、五档H、三个种子、两个rank。导出的压缩Chrome traces用于计算CUDA kernel/memcpy/memset活动区间的并集及空白区间；先对同一种子的两GPU平均，再汇总三个种子。它诊断初始阶段，有插桩开销，不替代正式吞吐或末轮精度；NCCL等待kernel也计作活动，空白比例不是SM利用率或所有低效等待的总量。

## 结果

每次双卡运行在`results/{数据集}/h{H}/seed{种子}/`，保存配置、划分、逐rank采样与元数据、实际本地更新`steps_rank0/1.csv`和同步轮次`rounds_rank0/1.csv`、20轮验证/漂移/墙钟、末轮测试和模型。`final.csv`、`seed_summary.csv`为主实验统计，`paired_comparison.csv`为同种子相对H=1的比较；`time_to_target_summary.csv`保留达标种子数，`profiles/`为90份独立时间线。

单卡参考保存于`results/single_gpu/`。`scaling.csv/scaling_summary.csv`为45对严格配对的吞吐/计算段加速比和效率，`ordinary_sgd_reference.csv`为9次普通SGD按H组织的45组任务参考。图1—11和配对统计统一位于报告“实测结果”。`results/step_metrics/`覆盖99次训练的753,480条rank/逻辑副本观测，已通过压缩CSV往返核验。

`distributed.py`实现参数打包、NCCL平均、解包和平均前漂移，`train.py`完成本地SGD与轮末平均，`metrics.py`按同步轮次复算。图1—4和图6直接比较灰色单GPU串行双副本与蓝色双GPU，依次展示通信量、吞吐/耗时、跨卡同步、漂移/准确率/显存及Time-to-Target。图5分四行展示单、双GPU的训练损失和验证准确率。三个种子用形状区分，粗线为均值。

单GPU设备内平均包含在完整训练耗时中，跨卡同步指标为0；两rank到达差只对双GPU定义。图7仅展示双GPU独立Profiler记录。图8—11将普通单模型SGD与同算法基线分别标识，指标与统计含义见报告。

`main_comparison.csv`保存90次同算法运行的主图数据，`convergence_plot.csv`保存对应20轮历史，`main_comparison_sources.json`记录配置和来源哈希，`figure_validation.json`核对曲线。绘图默认读取所选目录的`single_gpu/`，也可用`--single-results-dir`指定。
