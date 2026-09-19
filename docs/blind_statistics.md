# 真实未见题准确率与显著性

本协议只服务于一个较窄的声明：项目外保管并解封评分的“真实未见题准确率”。它不是证明算法有效、比较两个系统或形成研究贡献的必要条件。公开基准、自建自动真值、结构/来源留出上的比较应使用 [`automated_benchmark.md`](automated_benchmark.md) 的路线，并按其任务范围报告效应大小与不确定性。

若确实要输出可解释为“项目外封存真实未见题准确率”的报告，则必须使用封存基准和解封后的独立参考评分。本页的门槛不能反向解释为其他统计比较都需要外部评审。

## 必须满足的门槛

1. `BlindManifest` 的全部案例都必须是 `provenance=external_real` 且 `split=unseen`。
2. 每个案例必须在清单中有参考答案摘要；题面、附件和答案不能进入公开清单或选模流程。
3. 每个系统/案例运行都要使用清单中的完整预算。运行失败默认按错误处理；也可以显式关闭该策略，此时缺失格会保持 `not_assessed`。
4. 参考评分必须通过解封令牌核验，且包含预先指定的主评分字段。评审者不能与系统版本相同。
5. 调用方必须明确提供 `independent_evaluation_attested=true`。这是责任声明，不是程序可以替人完成的事实证明。
6. 案例数必须达到预注册的 `min_cases`，默认 20；正式研究应在方案中提前规定样本量、题目抽样范围、指标和显著性水平。

## 统计方法

`core.blind_statistics.assess_blind_accuracy` 将主评分按阈值转换为二元成功，输出每个系统的 Wilson 95% 区间。基线与处理系统使用同一题目配对，差异采用双侧精确 McNemar 检验，并用固定种子的 bootstrap 给出配对准确率差区间。只有所有门槛满足、`p < alpha` 且差异区间下界大于 0 时，状态才为 `statistically_significant`；否则为 `assessed_not_significant` 或 `not_assessed`。

```python
from core.blind_statistics import assess_blind_accuracy

report = assess_blind_accuracy(
    manifest, runs, unlocked_scores,
    baseline_system="registry-v1",
    treatment_system="open-search-v2",
    primary_score="numerically_correct",
    min_cases=30,
    independent_evaluation_attested=True,
)
```

网页端对应 `POST /api/research/blind-accuracy`。接口只接受已封存清单、运行账本和解封后的评分，不接受评估回调，也不会替调用方下载题面或参考答案。

命令行也可以在解封评分文件准备好后运行：

```powershell
python scripts/blind_benchmark.py statistics `
  --manifest benchmarks/public/manifest.json `
  --runs benchmarks/runs/baseline/*.json benchmarks/runs/treatment/*.json `
  --scores benchmarks/scores/*.json `
  --baseline-system registry-v1 --treatment-system open-search-v2 `
  --min-cases 30 --independent-evaluation-attested `
  --output benchmarks/reports/statistics.json
```

正式实验应把两套系统的运行文件和解封评分文件分别保存，避免把一个系统的评分误接到另一个系统。

## 如何解读

- `accuracy_claim=conditional_real_unseen`：可以在报告中写“在这批独立封存真实未见题上，准确率为……”，并同时给出题目范围、样本量、失败处理和评分协议。
- `statistically_significant`：只能说明在预先定义的这批配对题目上，处理系统相对基线的差异达到指定检验阈值；不是所有数学建模题都更好，也不是正确性定理。
- `assessed_not_significant`：已完成合规评估，但证据不足以支持显著改进。
- `not_assessed`：封存、独立评分、样本量或覆盖条件不满足，不能写准确率或显著性结论。

该模块不能替代真实题目的独立抽样，也不能证明参考评分本身没有偏差。正式发布前，应由项目外评审者保管解封令牌和参考解，并公开清单摘要、预注册方案、运行哈希和完整统计输出。
