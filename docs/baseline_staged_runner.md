# 基线优先与分阶段验证

`run_baseline_then_search` 将基线报告固定放在候选搜索之前，并区分两种模式：

- `exploration`：可以只有低成本阶段，结果只能作为探索结果。
- `strict_validation`：必须声明且执行一个最终确认阶段，候选仍需独立证据门才能作为最终结论。

超时、资源失败和预算不足不会被当作反例；空候选明确返回 `baseline_only`。该模块不执行不受信任代码，调用方仍需传入受监督 evaluator。
