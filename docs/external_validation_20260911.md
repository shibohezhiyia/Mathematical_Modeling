# 外部验证记录（2026-09-11）

## 目的

验证数据接入、留出预测、独立评分和跨任务比较链路是否能在项目外部来源上完整运行。这里的“外部”指数据不由本项目生成；它不等同于数学建模竞赛的真实未见题或专家评分。

## 固定协议

- 来源：UCI Machine Learning Repository，目录和下载地址固定在 [`examples/external_dataset_catalog.json`](../examples/external_dataset_catalog.json)。
- 数据集：Wine Quality（红/白两个案例）、Concrete Compressive Strength、Bike Sharing、Air Quality、Airfoil Self-Noise，共 6 个案例。
- 切分：固定种子 `20260910`，80/20；时间序列按时间顺序留出，其他案例固定随机留出。
- 比较：Ridge（基线）与 ExtraTrees（处理臂）；目标列先从训练流程中移除，并对 Bike 的派生列实施泄漏护栏。
- 评分：独立 Python 进程只接收测试真值和预测向量，计算 MAE/RMSE/R²；源文件和切分保存 SHA-256。
- 冻结：协议 `external-data-v1` 的源码/依赖/预算清单 digest 为 `24ecde9fdf20bc3def69e8fdc2c60f05737fcf2dd6bbcb6dcf449fb6455d24c5`，冻结文件位于本地 `data/reports/external_data_freeze.json`。

复现：

```powershell
python scripts/fetch_external_datasets.py --extract
python scripts/run_external_data_benchmark.py
python scripts/run_external_data_benchmark.py --seed 20260911 --output data/reports/external_data_benchmark_seed20260911.json
```

## 实际结果

| 指标 | 结果 |
| --- | ---: |
| 案例数 | 6 |
| 完成/失败 | 6 / 0 |
| 处理臂平均相对 RMSE（ExtraTrees/Ridge−1） | -0.263771 |
| 中位数相对 RMSE | -0.163848 |
| bootstrap 95% CI（任务级） | [-0.431034, -0.098282] |
| 精确双侧符号置换 p 值 | 0.0625 |

为检查单次切分偶然性，又以独立种子 `20260911` 重跑同一预注册协议：6/6 案例完成，平均相对 RMSE 为 `-0.264882`，精确双侧符号置换 `p=0.0625`。这只是切分敏感性复核，不能把重复使用公开数据当作新的独立题目。

处理臂在 6 个案例中的 RMSE 变化分别为 -15.35%、-16.91%、-56.40%、-15.86%、+2.25%、-55.99%。Air Quality 是处理臂退化的反例，说明系统应保留模型分歧而不能只展示平均值。

## 如何解释

这份结果证明了“公开外部数据 → 固定留出 → 独立评分 → 反例保留”的工程链路可运行，并提供了可复查的改善/退化证据。它不能证明：

1. 对真实未见数学建模题的准确率；
2. 相对当前系统的统计显著改进（任务数仅 6，`p=0.0625 > 0.05`）；
3. ExtraTrees 在所有问题上优于基线；
4. 模型能得到竞赛评委认可的数学论证。

Airfoil Self-Noise 的 UCI 官方页面说明该数据含 1503 个样本、5 个物理特征（含 Hz、m、m/s 等单位）和 dB 目标，适合检查带量纲数据是否仍能进入统一预测流程：[UCI Airfoil Self-Noise](https://archive.ics.uci.edu/dataset/291/airfoil%2Bself-noise)。UCI/OpenML 的公开数据和标准化基准可用于可复现方法比较，但不提供封存竞赛题的独立评委分数；如需该结论，必须另行获取组织方或盲评专家的评分。
