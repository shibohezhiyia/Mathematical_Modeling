# 独立未见题封存前的登记

`core.holdout_intake.validate_holdout_intake` 是可选的项目外封存评测元数据检查。只有研究要声称 `conditional_real_unseen` 时才需要这条流程；普通的公开基准、自动真值、结构留出和同预算算法比较不需要外部出题人或评审人。它不读取题面内容，只检查：

- 至少 20 道题、至少 4 个题型族（可按预注册方案调整）；
- 题面、附件和参考解均有 SHA-256 承诺；
- 出题人和评审人独立声明，以及每题的角色承诺摘要；
- 开发冻结摘要和评分量规摘要。

在选择这条可选评测臂时，通过后再由项目外评审者在仓库外逐题运行 `python scripts/blind_benchmark.py seal`。登记结果仍不能替代盲测清单、解封评分和统计检验；要发布 `conditional_real_unseen` 结论，必须经过 [真实未见题统计评估](blind_statistics.md)。

也可以用命令行校验一份只含哈希和角色承诺的登记文件：

```powershell
python scripts/validate_holdout_intake.py D:\sealed-benchmarks\holdout-intake.json
```

网页端对应 `POST /api/research/holdout-intake`。`_min_cases` 和 `_min_families` 只用于本次校验，不会写入登记结果。

封存后用 `POST /api/research/holdout-bind`（或 Python 的 `bind_holdout_intake`）逐题比较登记哈希、题型、附件和答案承诺与 `BlindManifest`；不一致时拒绝进入统计流程。

命令行等价用法：

```powershell
python scripts/bind_holdout.py D:\sealed-benchmarks\holdout-intake.json D:\benchmarks\public\manifest.json
```

推荐流程：

```text
独立出题/评审 → holdout_intake (元数据) → seal (哈希清单)
→ scan (泄漏检查) → 两套系统同预算运行 → 独立解封评分 → blind_statistics
```
