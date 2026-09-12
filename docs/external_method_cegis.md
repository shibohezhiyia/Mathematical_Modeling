# 外部方法 CEGIS 执行桥

`core.external_method_cegis` 将已存在的类型化 `pde_find`、`ude`、`ude_neural`
和 `llm_sr` 运行器接入统一的 `compile → evaluate → diagnose → patch → replay`
循环。候选只能是 JSON 方法负载，不接受源码、路径、模块或命令字段。

每个案例必须提供 `max_metric`（例如 PDE 的 `validation_rmse` 或 UDE 的
`holdout_rmse`）。因此“执行成功”不会自动成为通过；超过独立阈值才记录反例，
运行器不可用或字段缺失则保留为 `not_assessed`。参数变异仅限稀疏阈值、对流开关、
岭系数、网络宽度和已声明搜索代数，所有候选都会重新经过下游类型和资源检查。

网页入口：`POST /api/research/method-cegis`。它是研究/开发执行桥，不是任意代码
沙箱，也不替代独立题目、独立参考解或数学证明；生产部署仍应将数值 worker 放入
`strict_os` 隔离环境。
