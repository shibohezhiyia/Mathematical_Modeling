# 外部未见题基准来源

项目已经把可用外部来源登记在 [`examples/benchmark_sources.json`](../examples/benchmark_sources.json)，由 `core.benchmark_sources.BenchmarkSourceRegistry` 校验。注册表是“来源目录”，不是自动下载器：运行时不会把网页内容、题解或会员内容偷偷复制进仓库。

如果目标是证明“真实未见题准确率”或配对显著性，请先看[评估来源审计](evaluation_sources.md)。来源目录解决“去哪里找题/方法”，来源审计才会明确哪些材料仍然不能作为独立金标准。

## 可直接获取的外部数据

项目提供了一个只记录来源、许可证、固定字节数和 SHA-256 的目录：[`examples/external_dataset_catalog.json`](../examples/external_dataset_catalog.json)。五个 UCI 数据集覆盖表格回归、物理量纲、时间序列、缺失值和季节性；其中 Air Quality 标注为 UCI research-only（商业用途排除），其余数据集为 CC BY 4.0。下载到被忽略的 `data/external` 后才能进入实验。下载器只允许 UCI/OpenML HTTPS 域名，采用临时文件原子替换，并在哈希或大小不符时拒绝落盘。

```powershell
python scripts/fetch_external_datasets.py --dry-run
python scripts/fetch_external_datasets.py --dataset uci-bike-sharing --dataset uci-air-quality --dataset uci-airfoil-self-noise --extract
```

这些数据可用于外部数据处理、结构搜索和预测误差的可复现实验；由于公开标签可见，不能将其留出分数称为“真正未见题准确率”，也不能替代项目外独立评审。真正的未见题统计仍需外部封存题面和独立评分。

运行自动外部数据评测：

```powershell
python scripts/run_external_data_benchmark.py
```

网页端也可调用 `POST /api/research/external-data-benchmark`；服务端固定使用项目目录中的目录和 `data/external`，只接受有界的 `seed`、`max_rows` 参数。

评测器使用固定种子和 80/20 划分；时间序列按时间顺序留出，其余任务使用固定随机划分。每个任务同时跑 Ridge 基线和 ExtraTrees 处理臂，报告测试集 MAE、RMSE、R²、源文件哈希和划分哈希；跨任务比较使用相对 RMSE（处理臂/基线−1），同时保留均值、中位数、bootstrap 区间和精确符号置换检验，避免不同单位直接相加或单个任务掩盖分歧。结果写入被忽略的 `data/reports/external_data_benchmark.json`。本次运行包含 6 个案例（Wine 红/白、Concrete、Bike、Air Quality、Airfoil），全部完成；处理臂平均相对 RMSE 为 -0.2638，95% bootstrap 区间为 [-0.4310, -0.0983]，但仅有 6 个公开任务时精确双侧置换 p=0.0625，不能宣称统计显著或真实未见题准确率。这能自动检查数据读取、目标识别、留出预测和模型比较链路，但公开标签仍只支持描述性外部证据。

本次外部验证的机器可读结果位于本地被忽略目录 `data/reports/external_data_benchmark.json`；复现时重新运行上面的下载和评测命令即可。报告保存每个源文件 SHA-256、训练/测试划分哈希、独立评分器版本和数据清理审计。Airfoil Self-Noise 的官方说明给出 1503 个样本、5 个物理特征和 dB 目标，可作为带单位的独立回归案例（[UCI 官方页面](https://archive.ics.uci.edu/dataset/291/airfoil%2Bself-noise)）。公开数据的标签由数据集提供者给出，不是竞赛评委或封存专家金标准；真正的未见建模题结论仍需预注册题目、冻结系统、独立评审和盲态评分。

报告还计算任务级 Bootstrap 95% 区间和配对符号置换检验的双侧 p 值；样本少时仍只标记为 `descriptive_public_labels`，不会把 p 值单独解释成通用显著性。

数据契约会把 `cnt` 的组成列 `casual`/`registered` 和行号 `instant` 从 Bike Sharing 的输入中剔除，并把 Air Quality 的 `-200` 目标标记整行剔除；每个任务的 `data_audit` 会记录实际删除的列和行数。旧版未做这两项审计的结果已被当前报告覆盖，不再作为性能证据。

监督学习主链路也执行保守的目标泄漏审计：目标的精确加和分解会在拟合前删除并写入 `feature_leakage_filter`；精确复制列只标记给可信度审计，不凭相关性自动删除可能合理的预测变量；连续整数行计数器会作为索引候选剔除。`dropped_feature_columns` 和对应审计字段可追溯每次处理。

```python
from core.benchmark_sources import BenchmarkSourceRegistry

registry = BenchmarkSourceRegistry.load("data/benchmark_sources.json")
inputs = registry.by_role("task_input")
rubrics = registry.by_role("rubric_material")
references = registry.by_role("reference_only")
```

命令行也可以按用途列出来源：

```bash
python scripts/list_benchmark_sources.py --role rubric_material
python scripts/list_benchmark_sources.py --role reference_only
```

## 哪些来源可以用

| 用途 | 已登记来源 | 可以做什么 | 不能做什么 |
| --- | --- | --- | --- |
| 题目输入 | COMAP 官方 2025 题面/数据页 | 构造任务、检查附件哈希、做无数据题测试 | 不能把题面文字当作参考答案 |
| 评审量规 | COMAP 规则、UMAP 46.3/46.4 Judges' Commentary | 抽取“问题分析、模型、数学方法、验证、敏感性、优缺点、表达”等评审维度 | 不能从奖项直接推导准确率 |
| 方法对照 | MM-Agent / MM-Bench 及其代码 | 预注册同预算外部比较 | 不能把论文报告分数和本项目分数直接混合 |
| 参考候选 | mathworks 归档、公开获奖队伍仓库 | 人工复核、生成模型假设、代码烟测 | 不能当独立 gold truth；许可需逐仓库确认 |

COMAP 的 2025 MCM/ICM 页面列出了六道题及 C/D 附件；规则页明确说明评审重点是思路、分析、建模方法和数学方法，并要求讨论测试、误差、敏感性/稳定性、优缺点和结论。因此这些来源适合建立 rubric，不适合伪造唯一数值答案。

## 推荐的最小可用评测组合

1. 用 `comap-2025-problem-index` 或具体 PDF 作为输入来源，运行前固定 SHA-256。
2. 用 `comap-2025-rules`、`comap-umap-46-3-commentary` 和 `comap-umap-46-4-commentary` 建立评分表。
3. 用 `core.rubric_review` 要求至少两名独立评审，分别评分合同正确性、模型/数学论证、验证稳健性、证据完整性和可复现性。
4. 用 `mm-agent-mm-bench` 做外部方法对照；保持任务集、预算、模型和随机种子预先锁定。
5. 用公开题解仓库只做 `reference_only` 分支。只有项目外封存的独立 rubric/参考解在解封后，才允许写入 `core.blind_benchmark` 的分数。

已登记的补充评分材料包括 [IMMC 官方指南](https://www.imc-impea.org/immc.php)、[SIAM GAI MM 量规](https://www.siam.org/media/n2tinywa/gaimme-2nd-ed-final-download-print-bw.pdf) 和 [Zenodo 建模过程数据集](https://zenodo.org/records/18462009)。它们分别用于独立评审流程、量规校准和分阶段评分研究，均不作为真实未见题金标准。

仓库附带的 [`examples/benchmark_rubric_mcm_icm.json`](../examples/benchmark_rubric_mcm_icm.json) 是一个可直接传给 `core.rubric_review.build_rubric` 的八维模板。它的分数是描述性评审结果；至少两名独立评审、分歧记录和必要时仲裁后，才可以进入封存基准的解封流程。

来源和评审可以进一步绑定成一个不可静默修改的 material plan：`core.benchmark_material.build_material_plan(...)` 会检查每个来源的用途角色，并生成 `plan_digest`；`validate_material_plan(...)` 在执行前校验摘要。这样，公开题解不能误进入题面输入或评分量规，题面来源也不能被误当作参考答案。

仓库当前封存基准位于 `workspace/sealed_benchmarks/2026-09`，公开清单只保存题面/附件哈希，不保存题面正文、参考答案或评测结论。清单包含 15 个 `external_real` 案例，覆盖 2023–2025 年 MCM/ICM 的优化、多表、网络、动力学、统计、风险和政策问题。

每个题面的权威来源按年份和题号映射到 COMAP 官方 PDF，例如：[2025 ICM F](https://www.contest.comap.com/undergraduate/contests/mcm/contests/2025/problems/2025_ICM_Problem_F.pdf)。COMAP 的竞赛说明明确指出，官方题目、规则和评审结果以 COMAP 页面为准；其结果页提供各题获奖结果，但不是逐题数学参考答案。[官方规则](https://www.contest.comap.org/undergraduate/contests/mcm/instructions.php)、[2025 结果页](https://www.contest.comap.org/undergraduate/contests/mcm/contests/2025/results/index.html)

## 评测边界

- 题面和附件来自官方竞赛材料，运行前已写入 SHA-256；启动时测试会重新计算并拒绝缺失或不匹配文件。
- 参考答案没有写入公开仓库，因此系统只能评测结构、执行、约束、稳定性和证据完整性；不能把“成功运行”当成竞赛得分。
- 要做独立正确性评估，需要项目外保存的专家 rubric/获奖解，并在选择和提示开发之后才解封。不能从获奖名单反推出数学正确率。
- MM-Agent 的 MM-Bench 是独立的 111 题研究基准，可用于外部方法对照；其论文明确将任务拆为问题分析、结构化建模、计算求解和报告四阶段。[MM-Agent / MM-Bench](https://arxiv.org/abs/2505.14148)
- `jshn9710/mathworks` 是可供人工复核的公开题目、附件和解法归档，但其内容不是本项目的独立金标准，且每个文件的许可必须单独确认。[mathworks archive](https://github.com/jshn9710/mathworks)
- COMAP 公开了 2025 MCM/ICM 的获奖结果和 Judges' Commentary 入口；这些材料适合构造专家 rubric，但奖项本身不是逐题数值真值。官方公告确认 2025 MCM 与 ICM 的 Judges' Commentary 已发布。[COMAP 公告](https://www.comap.org/blog/news-announcements)、[2025 结果页](https://www.contest.comap.org/undergraduate/contests/mcm/contests/2025/results/index.html)
- COMAP 的 UMAP 46.4 页面列出 2025 MCM A/B/C 的 Judges' Commentary；UMAP 46.3 页面列出 2025 ICM D/E/F 的 Judges' Commentary。实际论文/评论可能需要 Mathmodels 会员，只在仓库保存入口和元数据。[UMAP 46.4](https://comap.org/membership/member-resources/item/umap-journal-46-4-winter-2025-edition)、[UMAP 46.3](https://comap.org/membership/member-resources/item/umap-journal-46-3-fall-2025-edition)
- 已登记的公开代码/解法仓库包括 [2025 MCM/ICM Problem C](https://github.com/david188888/2025-MCM-ICM) 和 [2025 ICM Problem D](https://github.com/CarmJos/MCM-ICM.2025.D)。它们只能作为候选模型和实现参考；仓库许可和内容授权必须逐项检查。

机器可读入口：`workspace/sealed_benchmarks/2026-09/public/manifest.json`。验证命令：

```bash
python -m pytest tests/test_external_benchmark_sources.py -q
```

封存目录默认被 `.gitignore` 排除，公开仓库没有题面正文和附件；没有本地封存目录时该测试会明确显示 `skipped`，不会把缺失语料误报为通过。
