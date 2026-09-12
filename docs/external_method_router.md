# 外部方法候选路由

`core.external_method_router.plan_external_methods` 将题目证据转换为有限候选列表。它只做路由，不导入外部仓库、不执行模型代码，也不把“有文献支持”当作正确性证据。

```python
from core.external_method_router import plan_external_methods

plan = plan_external_methods({
    "numeric_data": True,
    "time_series": True,
    "spatial_grid": False,
    "known_dynamics": True,
    "noise_expected": True,
})
```

路由规则是保守的：

- 时间序列可产生 SINDy；有噪声时额外产生 weak-form SINDy。
- 空间网格可产生 PDE-FIND/导数库候选。
- 已有机理右端项时才产生 UDE 修正候选。
- LLM-SR 默认仅为 `proposal_only`，只有显式允许外部仓库才会改变执行状态。
- 未提供依赖状态时返回 `not_assessed`，不会假设 PySINDy、JAX 或其他包已安装。

每个候选仍必须进入类型、单位、来源、资源和安全准入门；路由结果不能直接作为求解结果或论文结论。

Web 端可先调用 `POST /api/research/method-plan` 获取候选计划，再决定是否启动完整研究。对于本地适配器，可调用 `POST /api/research/method-execute` 执行单个受限方法，或将预注册 `manifest` 与任务列表提交到 `POST /api/research/method-compare` 执行配对比较；这些接口都不触发第三方安装或网络下载。`llm_sr` 在运行接口中始终返回 `proposal_only`。

完整 `MathModelingAssistant.run(...)` 也会在 `specialized_results.external_method_plan` 返回同一份保守计划。时间序列和空间网格只依据数据画像中的明确证据触发；缺失值不会被误判为噪声，因此不会自动开启 weak-form 路径。

本地执行结果会附带五项 `execution_readiness`。当前数据来源和单位没有随请求明确传入时，结果会保持 `not_assessed`，即使数值运算成功也不能直接进入最终结论或 Pareto 排名。

若调用方同时提供 `applicability_boundary` 和 `assumptions`，适配器还会生成有限的 `conclusion_certificate`，其中包含验证段残差和反例；缺少边界或元数据格式错误时证书保持 `not_assessed/blocked`，不会丢弃数值运行记录。

如需做正式方法比较，使用 `build_external_method_manifest` 固定五个实验臂（本地 SINDy、weak-form、PDE 特征库、UDE、LLM-SR 提议态）、任务集合、预算和最终测试指纹，再调用 `run_external_method_comparison`。每个任务的 `payloads` 按方法名提供开发段输入；运行器以同一任务网格逐格执行本地适配器，并保留缺失输入、提议态和未实现方法的失败行。

运行器输出的是描述性比较记录，并明确标注 `primary_metric=validation_residual_rmse` 与 `score_direction=lower_is_better`：SINDy/weak-form/UDE 的完成分数取留出残差均值，越低通常越好；PDE 特征库仍是规划态，LLM-SR 仍是提议态。失败始终留在分母，结果不能解释为真实未见题正确率、因果证明或统计显著性。需要配对效应时，再将结果交给 `summarize_paired_comparison(..., score_direction="lower_is_better")`，并使用独立 gold truth 与最终测试集。
