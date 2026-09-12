# 候选硬证据门

`gate_candidates` 先检查量纲、约束、类型等硬证据，再对合格候选做 Pareto 比较。硬失败、待评估和资源耗尽不能被更好的拟合分数抵消；`classify_failure` 只记录分类，不凭错误码臆测根因。

候选显式声明 `execution_status` 为 `executed`、`runnable`、`implemented`、`executable` 或 `solver_ready` 时，还必须提供完整的 `execution_readiness`（类型、单位、来源、资源、安全五项）。没有这五项证据的运行结果只能保持 `not_assessed`，不能进入 Pareto 排名；纯提议候选不受此规则误伤。
