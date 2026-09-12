# 真实未见题基准

“真实未见题”不是把开发题换一个编号，也不是随机生成一份合成数据。它必须满足：题面、附件和参考解在系统选型、提示词开发、代码搜索和参数调整期间都不进入公开工作区；运行结束后才由独立评测者解封计分。

项目提供 `core/blind_benchmark.py` 和 `scripts/blind_benchmark.py` 作为最小可审计协议：

1. `seal` 只记录题面、附件和参考解的 SHA-256、文件大小与解封承诺，不复制内容。
2. `scan` 在运行前扫描公开仓库，检查案例编号、完整文件摘要和仓外保护文件是否泄漏；对可解码文本还做有界归一化词片段重叠筛查，不能替代语义相似度审查。
3. `record` 强制使用清单中冻结的时间、内存、API、候选数量和随机种子，记录完成、失败、阻塞或超时；预算不一致会拒绝写入。
4. `score` 只有持有仓外令牌并匹配参考解摘要时才能解封。解封前的报告明确为 `not_scored`，不会把未运行或未评分当成失败/零分。
5. `report` 输出运行覆盖、失败状态和解封后的分数均值，同时标记案例来源。合成夹具永远不会被报告为真实未见题证据。

## 推荐目录

公开仓库只放：

```text
benchmarks/public/manifest.json
benchmarks/runs/<system-version>/<run-id>.json
benchmarks/reports/<run-id>.json
```

题面、附件、参考解和令牌放在仓库外的只读目录，例如：

```text
sealed-benchmarks/2026-09/
  questions/
  answers/
  unlock-tokens/
```

不要把令牌文件、题面目录或答案目录加入 Git。公开清单只含哈希，因此任何人都可以验证“运行时使用的对象是否就是封存对象”，但不能从清单恢复题面。

## 命令示例

```powershell
python scripts/blind_benchmark.py seal `
  --case-id unseen-001 --family optimization `
  --statement D:\sealed-benchmarks\questions\unseen-001.txt `
  --answer D:\sealed-benchmarks\answers\unseen-001.json `
  --token-file D:\sealed-benchmarks\unlock-tokens\unseen-001.token `
  --output benchmarks/public/unseen-001.json

python scripts/blind_benchmark.py scan `
  --manifest benchmarks/public/unseen-001.json `
  --project-root . `
  --exclude benchmarks/public/unseen-001.json
```

多个案例应由一个冻结清单统一管理，且运行命令的预算参数必须与清单一致。开发题只能用于冻结系统；不能读出未见题后反复修改提示、注册表或求解器再重新提交。若必须修改系统，应增加系统版本并重新从未解封清单运行。

## 如何判断结果是否真的有意义

一次运行报告只能证明执行状态，不等于数学正确。至少需要同时保留：

- 清单摘要、系统版本、依赖版本、随机种子和固定预算；
- 每个案例的结构/编译/数值/验证失败归因；
- 参考解解封后的约束满足、数值误差、稳定性和人工复核分数；
- 解封前没有进入搜索过程的证据（仓库泄漏扫描和运行日志）。

只有外部真实案例全部满足封存、独立解封和审计条件后，才可以谈“真实未见题成功率”。本目录中的 `synthetic_fixture` 仅用于测试协议，不可充当泛化证据。

如需进一步判断两套系统的准确率差异是否达到预设显著性水平，使用 [真实未见题统计评估](blind_statistics.md) 中的 `assess_blind_accuracy`。它要求全部案例为外部真实 `unseen`、参考评分已解封且调用方完成独立评审声明；不满足条件时只返回 `not_assessed`。
