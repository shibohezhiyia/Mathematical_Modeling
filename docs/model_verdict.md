# 模型判决书

`build_model_verdict` 是研究结果与网页/论文输出之间的保守边界。它不会把排序分数、有限留出测试或模型调用的自评包装成概率或证明。

- `approved` 只能由调用方显式提供，且仍注明需要独立确认；
- `conditional` 表示当前证据下可继续使用的候选，不是最终获准；
- `unresolved` 覆盖缺少输入、执行错误、预算耗尽和不可辨识；
- `rejected` 只记录明确的反例或硬约束失败。

报告同时保存四层不确定性、共同决策、候选分歧、反例、假设、证据引用和下一步问题。没有候选共识时，`minimum_common_conclusion.status` 为 `not_established`，不会强行输出单一推荐。

数值求解器可以传入 `numerical_stability` 摘要。判决书只保留容差比较状态、成功/失败次数和指纹：`stable_on_tested_tolerances` 只映射为 `tested_not_falsified`，`unstable` 映射为 `counterexample_found`；两者都不是全局数值误差证明，也不会自动把候选提升为 `approved`。

若存在 `uncertainty_propagation`，判决书还会接收语义、结构、参数和数值层的经验方差组件，并保留统一单位、记录数及重构方差摘要。原始评估行不会进入判决书，分解结果也不等于概率后验或置信区间。
