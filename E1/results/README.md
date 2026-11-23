# E1 结果

正式运行采用 CUDA/NCCL、三个数据集、S/M/L、三个种子和单/双卡，共54次，每次20轮。实验报告见 [实验报告.md](实验报告.md)，使用方法见 [E1/README.md](../README.md)。

- `mnist/`、`fashionmnist/`、`cifar10/`：每个模型、卡数和种子各有独立目录，保存原始逐rank计时、训练验证记录、末轮模型与最终测试。
- `environment.json`、`host_context.json`、`provenance/`：正式训练环境、虚拟化主机信息、运行源码及哈希。
- `analysis_sources.json`、`analysis_source/`：正式归档的分析源码快照与哈希。现行绘图入口见 [复现说明](../README.md#运行)。
- `preflight.json`：正式运行前10项检查结果，涉及模型计算时使用GPU。
- `run_plan.log`、`training.log`：运行计划和正式训练输出。
- `final.csv`、`seed_summary.csv`：绘图时从54次原始结果汇总的指标、均值、样本标准差与原始范围。
- `paired_scaling.csv`：先按同一种子计算双卡/单卡吞吐比与扩展效率。
- `model_sizes.csv`：九种数据集/模型组合的参数量、梯度载荷和前向矩阵运算量。
- `figs/`：图1为更新时间分段；图2为同步占比与同步/计算时间比；图3为吞吐和弱扩展效率；图4为梯度载荷与双卡耗时。
- `validation.json`、`validation.log`：54次记录核验与54个模型独立GPU复测均通过；`--records-only`另写 `records_validation.json`。
- `figure_validation.json`：已确认四张指标图的三种子观测均位于显示范围内。
- `original_gpu_files_sha256.json`、`completion.json`：567份原始文件的哈希与54次运行、54次GPU独立复测的完成记录。

单、双GPU的三个模型规模均独立训练，报告中的统计与图片可以从本目录的原始记录复算。

新实验默认写入 `reproduce/`，核验和绘图可选独立结果目录。具体命令见 [复现说明](../README.md#运行)。
