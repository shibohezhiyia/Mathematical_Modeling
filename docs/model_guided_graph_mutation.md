# 模型引导的受限数学图变异

`GraphPatchGenerator` 将本地小模型或兼容 API 限制为 **IR patch 提议者**。模型不会直接生成 Python、执行程序、改变题面或宣布候选正确。

调用链为：

1. 数值评价器产生状态、结构化违反原因和有限反例；
2. `minimal_patch_feedback` 删除堆栈和无关字段，只保留有界诊断；
3. 模型返回最多 4 个 JSON patch；
4. `apply_graph_patch` 重新检查父哈希、改动数、符号、输出、类型、单位、来源和图依赖；
5. 合法候选进入原有 CEGIS 队列，重新执行全部检查；模型评分不参与最终判定。

```python
generator = GraphPatchGenerator(config)
result = GraphSearchSession(contract, experiment).run(
    [initial_graph.payload()],
    model_patch_generator=generator,
)
```

自动循环只接受 `HttpSemanticBackend`，因为它具备请求超时。任意 Python 回调可以用于离线提议测试，但不能进入搜索循环；这避免一个卡住的回调绕过墙钟预算。进入模型调用前，剩余墙钟必须不小于配置超时。模型调用次数和候选数由服务端 `SearchController` 计数，失败调用同样消耗次数。

当前边界：

- 只接入标量图 CEGIS；ODE、优化和多表图还未接入；
- JSON Schema 是传输提示，真正的授权边界仍是类型化 IR 校验；
- HTTP 超时不是操作系统权限隔离；
- 反例是有限域内见证，候选通过仍仅表示“可进入冻结留出确认”，不是现实正确或数学证明；
- 本地网页“数学图实验”可以显式勾选模型变异，并复用研究助手当前连接设置；公开模式仍禁用整个实验室；
- 尚未做真实 API 成本与质量基准。
