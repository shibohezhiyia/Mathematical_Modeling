# 流式任务与联表计划

`estimate_join_cardinality` 根据两侧键频数计算多对多输出上界，爆炸时只返回 `needs_semantic_aggregation`，不会擅自聚合。`plan_external_state_operation` 强制排序、联表、窗口等跨块任务声明外存/状态协议，并记录任务所需列投影；这层是准入计划，具体 spill/外存执行器仍需单独验证。
