# E5结果

45次双GPU/NCCL主实验、45次单GPU串行双副本Local SGD和9次单GPU普通SGD参考均已完成，每次20轮，覆盖三个数据集和三个种子。99个末轮检查点已在CUDA上独立复测。报告见[实验报告.md](实验报告.md)，运行入口见[上级README](../README.md)与[单卡说明](../single_gpu/README.md)。

`{数据集}/h{H}/seed{种子}/`保存720个双卡原始文件：实际每次本地SGD的`steps_rank*.csv`、H步同步轮次的`rounds_rank*.csv`、配置、划分、采样哈希、运行元数据、显存、20轮验证、末轮测试和模型。`environment.json`、`provenance/`保留环境和训练来源，`preflight_step_logging.json/log/xml`记录19项主实现与单卡联合检查。

`single_gpu/`保存585个单卡原始文件。串行双副本的`local_steps.csv`记录每次SGD，`rounds.csv`记录完整轮次；普通SGD记录于`steps.csv`。单卡跨卡归约次数和载荷为0，设备内平均另计。两种基线与按H组织普通SGD参考的规则均在“实测结果”中说明。

`final.csv/seed_summary.csv`为主实验指标，`paired_comparison.csv`为同种子相对H=1的比较，`time_to_target_summary.csv`保留首次验证达标信息。`scaling.csv/scaling_summary.csv`保存45对严格同算法的吞吐加速比、计算段加速比及并行效率；`ordinary_sgd_reference.csv`保存9次普通SGD按H整理的45组任务参考。

`step_metrics/rank_steps.csv.gz`包含753,480条实际rank/逻辑副本更新观测，`global_units.csv.gz`包含175,440个同步轮次或普通SGD更新单元。`audit.json`核对与原始运行汇总的一致性，`roundtrip_validation.json`重新读取压缩CSV检查公式、事件、行数与来源。

90份独立双rank Profiler时间线位于`profiles/`，不混入正式训练计时。`profile_activity.csv/profile_seed_summary.csv`保存逐rank与两卡先平均的记录，`profile_validation.json`复算全部时间线及64/H通信kernel次数，`profile_traces_sha256.json`保护原始压缩文件。

`validation.json`与`single_gpu/validation.json`记录45个双卡和54个单卡模型的独立CUDA复测。`gpu_control_comparisons.json`记录45对串行与双卡末轮模型的比较。`figure_validation.json`核对七张主图，`scaling_figure_validation.json`核对四张单、双GPU对照图，`scaling_report_validation.json`核对报告表格、公式和链接。

双卡与单卡原始文件分别由`original_gpu_files_sha256.json`、`single_gpu/original_gpu_files_sha256.json`保护。

图1—6采用45次同算法单GPU串行双副本与45次双GPU结果，数据为`main_comparison.csv`；图5的20轮历史为`convergence_plot.csv`。图4展示漂移、准确率和每卡显存，图6比较达标时间。`main_comparison_sources.json`记录来源哈希，`figure_validation.json`核对种子与均值。两rank到达差及图7的`profile_plot.csv`仅涉及双GPU；图8—11将普通SGD参考与严格基线分别标识。全部图表连续编号，位于报告“实测结果”。
