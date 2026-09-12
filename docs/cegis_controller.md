# 通用 CEGIS 控制器

`core.cegis_controller.run_cegis` 是跨题型的候选搜索控制层。它不理解 ODE、优化或统计的具体数学，只约束候选队列、反例反馈、候选哈希去重、修复次数和停止状态。

- `evaluate(candidate)` 只能返回 `pass`、`fail` 或 `not_assessed`，并可附有限的 `violations`；
- `mutate(candidate, feedback)` 只能提出新候选，不能改题面、阈值或评价器；
- `feedback` 会带有有界、脱敏的 `counterexample_archive`，其中只保留候选哈希、轮次、失败码、原因和见证 ID，帮助变异器避免重复旧反例；
- 只有 `pass` 且没有违反记录的候选才进入 `accepted_candidate_hashes`；
- 该列表表示 `tested_not_falsified`，不是数学证明，也不替代独立留出验证；
- 评价器异常、预算耗尽和候选不足分别保留可审计状态，不把最后一个未检查候选当答案。
- 每条记录保留 `parent_hash`，返回的 `lineage` 只描述候选版本关系，不把父候选的通过状态继承给子候选。反例归档有独立上限，超出部分计入 `dropped_counterexamples`，不会静默宣称完整历史。

进程、内存和网络权限仍必须由外部受控 worker 提供；控制器不会把任意回调变成安全沙箱。
