# E3结果

E3比较固定全局Batch=256的四种Micro-Batch/累积实现，每种均设置相同全局样本窗口和累积次数的单GPU基线。三个数据集、三个种子、每次20轮，单、双GPU各36次，共72次训练，72个末轮模型均完成CUDA独立复测。报告见[实验报告.md](实验报告.md)，使用方式见[E3/README.md](../README.md)。

各数据集的`b{micro}_k{累积}/seed{seed}/`目录保存完整记录、样本顺序哈希、末轮模型及测试。`environment.json`与`provenance/`保存环境和实际代码；`preflight.log/xml/json`记录两组联合14项前置检查，其中E3为8项。

`final.csv`、`seed_summary.csv`、`paired_relative.csv`、`figs/`由训练记录生成；`implementation_differences.csv`记录与128×1的末轮参数及测试差异。`validation.json/log`保存记录核验与GPU复测结果。

`numerical_equivalence.csv/log`记录真实完整窗口、尾批的FP32单步差异与GPU FP64参考检查，`relu_numerical_diagnostic.json`记录批形状引起的ReLU符号差异，`baseline_agreement.json`核对E2/E3九对相同配置的基线。诊断数据不计入正式性能统计。`original_gpu_files_sha256.json`保护504个双卡运行文件，单卡文件由`single_gpu/`中的同名清单保护；`figure_validation.json`和`report_crosscheck.json`记录图表、统计核验。

`single_gpu/`保存36次单GPU训练的配置、逐步计时、验证测试与检查点。

`scaling.csv/scaling_summary.csv`保存36对同种子加速比、计算段加速比、并行效率与质量差异，`scaling_validation.json`核对训练和测量窗口。图1—7及配对结果统一位于报告“实测结果”。`step_metrics/`提供430,560条rank观测和287,040次全局更新的耗时、吞吐、同步及来源。

图1—4直接比较相同全局256、同K的单、双GPU，横轴标双卡每卡micro×K，单卡micro为其两倍。`main_comparison.csv`含72次运行，`main_comparison_sources.json`记录来源和配对，`figure_validation.json`核对种子、均值和图片数据。
