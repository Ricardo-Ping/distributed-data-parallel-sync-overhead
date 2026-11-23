# E2结果

E2固定双GPU、累积1次，扫描每卡Batch Size，并为每档设置匹配相同全局Batch的单GPU基线。三个数据集、三个种子、每次20轮，单、双GPU各45次，共90次训练，90个末轮模型均完成CUDA独立复测。报告见[实验报告.md](实验报告.md)，使用方式见[E2/README.md](../README.md)。

各数据集的`b{batch}_k1/seed{seed}/`目录保存配置、划分、逐rank采样与耗时、逐轮验证、末轮模型和测试。`environment.json`与`provenance/`记录实际环境和运行代码；`preflight.log/xml/json`记录两组联合14项前置检查，其中E2为6项。

`final.csv`、`seed_summary.csv`、`paired_relative.csv`与`figs/`由绘图程序从训练记录生成。`validation.json/log`记录原始指标核验和GPU独立复测。

`single_gpu/`保存45次单卡训练的配置、模型、逐步计时和核验；`scaling.csv`、`scaling_summary.csv`与`scaling_validation.json`保存配对加速比、效率及复算。全部统计图按图1—7连续编号，统一位于报告“实测结果”。双卡的630个运行文件由`original_gpu_files_sha256.json`保护，单卡文件由`single_gpu/`中的同名清单保护；`report_crosscheck.json`核对报告中的统计数值。

`step_metrics/rank_steps.csv.gz`、`global_units.csv.gz`覆盖全部90次运行，记录每步耗时、样本吞吐、同步事件和载荷；`run_index.csv`、`audit.json`保存来源与公式检查。原始记录由哈希清单保护，指标导出不覆盖测量文件。

图1—4直接比较相同全局Batch的单、双GPU。`main_comparison.csv`含90次运行，`main_comparison_sources.json`记录配置、工作量与源文件哈希；`figure_validation.json`核对三个种子、均值、坐标范围和图片哈希。

新实验默认写入 `reproduce/`，核验和绘图可选独立结果目录。具体命令见 [复现说明](../README.md#运行入口)。
