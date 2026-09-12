# 分级候选评估

`core.staged_evaluation` 提供一个与具体求解器无关的候选调度器。它把结构检查、低成本筛选、高保真求解和独立确认串成多个阶段，但不替调用方决定数学结论。

```python
from core import StageSpec, run_staged_evaluation

result = run_staged_evaluation(
    candidates,
    evaluator,
    [
        StageSpec("static", 1, keep_fraction=0.5, min_survivors=3),
        StageSpec("fit", 5, keep_fraction=0.5, min_survivors=2),
        StageSpec("confirm", 20, min_survivors=1, confirmation=True),
    ],
    max_total_budget=200,
)
```

`evaluator(candidate, stage)` 只需返回包含 `status` 的映射，可选返回 `score` 和 `evaluations_used`。只有调用方明确列入 `hard_failure_statuses` 的状态才会被当作反例淘汰；执行错误、超时或预算耗尽的候选会保留为 `unresolved`，避免把“没来得及验证”误报为“数学错误”。

确认阶段必须是最后一个阶段，且调度器会从总预算中预留 `per_candidate_budget × min_survivors`，除非调用方明确不设置总预算。输出包含每次评估、阶段幸存者、预算消耗和可机器读取的策略说明。它仍然不是安全沙箱，也不是独立证据：实际执行应放在受监督的 worker 中，最终结论仍需通过模型审计、反例测试和证据门。

