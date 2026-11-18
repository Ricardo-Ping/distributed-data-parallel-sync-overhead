# E4结果

E4比较五档K下的两种双GPU同步方式，并为每档K设置匹配的单GPU基线。三个数据集、三个种子、每次20轮，共90次双GPU与45次单GPU训练，135个末轮模型均完成CUDA独立复测。报告见[实验报告.md](实验报告.md)，使用方式见[上级README](../README.md)。

`{数据集}/k{K}_{方式}/seed{种子}/`保存1260个双卡原始记录与模型文件；`environment.json`和`provenance/`记录环境与代码；`preflight.log/xml/json`保存联合22项前置检查，其中E4为13项。`final.csv`、`seed_summary.csv`、`paired_comparison.csv`为双卡逐种子指标、分组统计和同K配对比；`validation.json/log`为原始复算和GPU独立复测，`report_crosscheck.json`、`figure_validation.json`为表格和图1—5的数据核验。双卡原始文件由`original_gpu_files_sha256.json`保护，单卡文件由`single_gpu/`中的同名清单保护。

单卡micro256、同K、同全局batch，45次训练的配置、逐步记录和模型位于`single_gpu/`；同K两种双卡方式共用一份单卡基线，形成90对比较。

`scaling.csv/scaling_summary.csv`保存90对吞吐加速比、计算段加速比、并行效率与质量差异，`scaling_validation.json`复算工作量和公式。图1—8及配对结果统一位于报告“实测结果”。`step_metrics/`包含348,900条rank观测、209,340次全局更新的耗时、吞吐、同步计数、载荷与来源。

图1—4展示灰色单GPU与蓝、橙两种双卡实现，同K共用一份单卡基线。`main_comparison.csv`含135次运行，`main_comparison_sources.json`记录配对与来源，`figure_validation.json`核对曲线。图5使用`sync_optimization_plot.csv`，只比较两种双卡同步策略。
