# 哪些来源可以支持“真实未见题”结论

仓库新增 [`examples/evaluation_sources.json`](../examples/evaluation_sources.json) 和
`core.evaluation_source_audit`，把来源审计与盲测统计分开。运行：

```powershell
python scripts/validate_evaluation_sources.py
```

网页端可通过 `GET /api/research/evaluation-sources` 查看同一份脱敏审计结果；接口只返回来源元数据，不会下载题面、附件或参考解。

## 当前核验结论

| 来源 | 适合用途 | 能否直接支持真实未见题准确率 |
| --- | --- | --- |
| ModelingAgent / ModelingBench | 最接近数学建模的公开对照和评分量规参考 | 不能；题目和数据公开 |
| MM-Agent / MM-Bench | 外部方法比较、阶段性回归 | 不能；公开 MCM/ICM 题目且不提供独立金标准 |
| Contest Modeling expert evaluation (2026) | 阶段式量规、自动评分与独立专家一致性方法 | 不能；公开竞赛题，适合作为评分协议参考 |
| COMAP 官方题目页 | 新题执行回归和题型覆盖 | 不能；公开题面，开放题也没有唯一数值答案 |
| IMMC 官方指南 | 三位教授独立评审流程参考 | 不能；题源公开，适合设计评审流程 |
| SIAM GAI MM 量规 | 数学建模评分维度和量规校准 | 不能；是量规材料，不是未见题集 |
| Zenodo 建模过程数据集 | 校准分阶段评分和评审一致性 | 不能；是课堂评估数据，不是封存竞赛题 |
| IMProofBench | 私有题、专家评分和防污染流程参考 | 不能直接用于本项目；它评估数学证明，不是建模 |
| FrontierMath | 私有/原创数学题的防污染流程参考 | 不能直接用于本项目；它不是建模任务 |
| FrontierScience | 专家量规和留出科学推理评估参考 | 不能直接替代建模准确率 |
| DrivenData 隐藏标签竞赛 | 外部预测准确率、代码执行和安全评测 | 不能替代开放式建模；必须按具体竞赛协议登记 |
| TML-Bench | 私有留出标签上的表格机器学习预测分数 | 不能替代专家评分的建模质量 |
| Kaggle 隐藏真值竞赛 | 隐藏 solution 文件和外部排行榜机制 | 只能作为具体竞赛的预测指标，不能当作建模金标准 |

因此，当前目录中**没有**一个现成公开下载、同时满足“数学建模领域匹配 + 私有留出 + 独立参考评分”的来源。公开来源可以帮助测执行可达性和方法差异，但不能把通过率写成真实未见题准确率。

## 已找到的第二条评测臂：隐藏标签预测

DrivenData、Kaggle 的代码竞赛以及 TML-Bench 提供了另一种可落地的外部指标：测试标签对系统隐藏，由平台或基准维护者返回预测分数。这适合检验数据读取、特征处理、预测和资源约束是否可靠；它回答的是“预测误差是多少”，不是“开放式数学模型是否合理”。项目中将它们登记为 `method_comparison`，不会进入专家建模准确率统计。

使用这条评测臂时仍需固定：竞赛/基准 ID、开发截止时间、数据访问权限、提交次数、主指标方向、失败计分和最终测试指纹。平台公开说明或论文只能证明评测机制存在，不能证明本项目已经获得该竞赛的分数，也不能把公开排行榜分数改写成真实未见题结论。

平台返回分数后，可用 `scripts/import_hidden_scores.py` 导入。输入只允许分数、任务 ID、评测器、指标和测试指纹，禁止携带预测值或真实标签：

```powershell
python scripts/import_hidden_scores.py `
  --scores external_scores.json `
  --min-tasks 5 `
  --output workspace/reports/hidden-label-summary.json
```

输出固定标记为 `descriptive`，后续如需比较两个系统，应使用预注册的配对比较协议；该摘要本身不会产生显著性或建模质量结论。

两名评审填写表格后，用私有映射汇总：

```powershell
python scripts/aggregate_review_forms.py `
  --packet workspace/review_packets/2026-09/packet.json `
  --mapping workspace/review_packets/2026-09/private_mapping.json `
  --rubric examples/benchmark_rubric_mcm_icm.json `
  --forms workspace/review_packets/2026-09/form_reviewer-1.json workspace/review_packets/2026-09/form_reviewer-2.json `
  --output workspace/reports/heldout-rubric-review.json
```

## 无法申请组织方时的可执行替代

仓库已经附带一批本地封存的 COMAP 历史题：
`workspace/sealed_benchmarks/2026-09/public/manifest.json` 当前有 15 题、6 个以上题型族，运行记录为 15/15 完成。这些题目相对于冻结后的提示和参数可以作为外部冻结留出集，但它们不是“从未公开”的题。

该清单目前没有参考答案摘要，报告中的 `scored_count=0` 是预期状态。你可以让两名没有参与开发的人只看系统输出，使用 `examples/benchmark_rubric_mcm_icm.json` 独立打分，再用 `core.rubric_review.aggregate_reviews` 汇总。这能得到“冻结留出集上的独立盲审质量分”，不能改名为真实未见题准确率；只有组织方或仓外持有人提供独立参考解并完成解封，才允许进入 `blind_statistics` 的准确率/显著性流程。

可用命令生成匿名评审包（`private_mapping.json` 必须留在仓库外）：

```powershell
python scripts/build_review_packet.py `
  --manifest workspace/sealed_benchmarks/2026-09/public/manifest.json `
  --runs workspace/sealed_benchmarks/2026-09/runs/run_records.json `
  --rubric examples/benchmark_rubric_mcm_icm.json `
  --result-root workspace/sealed_benchmarks/2026-09/runs `
  --output-dir workspace/review_packets/2026-09
```

## 可发表的最小方案

需要项目外评审者建立至少 20 道未公开建模题（建议按优化、动力学、统计、网络、空间和无数据机理分层），在开发冻结前保存题面、附件和评分量规；开发者只看到哈希承诺。每个系统用同一固定预算运行，评审者解封后按预注册量规独立评分。

然后用 [`core.blind_statistics.assess_blind_accuracy`](blind_statistics.md) 计算 Wilson 区间、配对精确 McNemar 检验和 bootstrap 差异区间。只有 `accuracy_claim=conditional_real_unseen` 才能报告“在这批封存题上的准确率”；`statistically_significant` 也只说明这批配对题上的差异，不是对所有未来题的定理。

来源级 `accuracy_eligible` 永远不会自动变成显著性结论；样本量、运行覆盖、独立评分和解封都必须在案例级再次通过。
