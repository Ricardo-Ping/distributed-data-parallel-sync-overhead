# E5单GPU参考

E5保留两种单GPU参考，避免把系统并行和算法变化混在同一个加速比中。

| 参考 | 一次运行的实际执行 | 用途 | 运行数 |
|---|---|---|---:|
| serial_local_sgd | 一张GPU、一个进程，串行执行两个逻辑模型；每副本batch128、独立momentum，每H步设备内平均参数 | 与双卡Local SGD严格配对的同算法基线 | 45 |
| ordinary_sgd | 一张GPU、一个模型，batch256，每步SGD | 常规单卡任务参考；H>1时与Local SGD算法不同 | 9 |

两个逻辑模型都完整位于同一CUDA设备；设备内平均单独计时，不执行NCCL，跨卡同步计数及字节均为0。平均模型参数，不平均或清空本地momentum；epoch末不足H也平均，与双卡协议相同。基础配置读取`../configs/`，实际Case与参考配置保存于每次运行的`config.json`。

每个数据集和种子的普通SGD只训练一次，按H重新组织计时窗口供比较；45组H参考值仍只来自9次真实运行。串行双副本按H逐一运行，不能用普通SGD的耗时替代。所有训练均20轮，每轮验证，末轮测试一次。

依赖安装和GPU检查见项目的[依赖安装](../../README.md#依赖安装)，训练前先[准备数据](../../README.md#数据准备)。单GPU参考各使用一张CUDA GPU，包含双卡任务的配对启动器需要两张可见GPU和NCCL。

在项目根目录执行。配对启动器交错安排双卡和单卡任务，性能采集逐次串行：

```bash
uv run python -m E5.single_gpu.paired_run --list
uv run python -m E5.single_gpu.paired_run --dataset mnist --resume
uv run python -m E5.single_gpu.paired_run --dataset fashionmnist --resume
uv run python -m E5.single_gpu.paired_run --dataset cifar10 --resume

# 只运行所选单卡参考时
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run python -m E5.single_gpu.run --role serial_local_sgd --list
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run python -m E5.single_gpu.run --role ordinary_sgd --resume

uv run pytest E5/tests E5/single_gpu/tests -q
uv run python -m E5.check_results
uv run python -m E5.single_gpu.check_results
uv run python -m E5.single_gpu.compare
uv run python -m E5.single_gpu.plot_results
uv run python -m E5.evaluate --checkpoint E5/results/single_gpu/mnist/serial_h4/seed1234/checkpoint.pt --split test --device cuda:0
```

已有完整结果用`--resume`跳过；未完成目录先归档。单卡结果在`../results/single_gpu/{数据集}/serial_h{H}/seed{种子}/`和`ordinary_sgd/seed{种子}/`。串行双副本`local_steps.csv`逐次记录实际SGD、样本数与吞吐，`rounds.csv`记录两个副本完整同步轮次；普通SGD的`steps.csv`保存每次更新。两种时间范围分别保留，不把轮次时间除以H伪造每个本地步。

重新实测到独立目录时，训练与分析需指向同一批结果：

```bash
uv run python -m E5.single_gpu.paired_run --dual-output-dir E5/results/new_dual --single-output-dir E5/results/new_single
uv run python -m E5.check_results --results-dir E5/results/new_dual
uv run python -m E5.single_gpu.check_results --results-dir E5/results/new_single --dual-results-dir E5/results/new_dual
uv run python -m E5.single_gpu.compare --results-dir E5/results/new_single --dual-results-dir E5/results/new_dual --output-dir E5/results/new_dual
uv run python -m E5.single_gpu.plot_results --results-dir E5/results/new_dual
```

`scaling.csv`保存45组严格同算法配对，`ordinary_sgd_reference.csv`保存9次普通SGD按H组织的45组任务比较。图8—11展示两个基线的吞吐、质量、显存和验证达标时间，未达标保留删失。全部定义、统计及图1—11统一位于[实验报告](../results/实验报告.md)的“实测结果”。
