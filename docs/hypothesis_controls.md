# 假设滑块与局部重算

`build_hypothesis_controls` 将假设参数编译为有限范围、步长、单位和受影响节点的控件 schema；`affected_nodes_for_control` 对图的下游闭包生成局部重算计划。该计划只用于 what-if 试验，不能绕过 IR/资源/证据门，实际使用应与全图重算做一致性对照。

## 实际预览执行

`POST /api/research/hypothesis-preview` 已把控件从“只生成计划”推进到受限的类型化子图预览：

1. 服务端重新校验控件范围、步长和 finite 数值；浏览器提交的结果不会被信任。
2. `parameter_id` 将控件绑定到显式参数，`version` 和 SHA-256 digest 记录本次假设分支。
3. 仅对 `affected_nodes_for_control` 的下游闭包生成重算计划；若提供 `mathmodel.graph/v1`，则由 `execute_primitive_graph` 在类型/量纲运行时内重新计算。
4. 返回 `preview`、`execution` 和策略字段；未绑定参数、未决机制、非法函数或资源错误不会被包装成数学结论。

示例请求（数值仍需用户确认）：

```json
{
  "controls": [{"id": "gain", "parameter_id": "k", "label": "增益", "min": 0, "max": 2, "step": 0.1, "default": 1}],
  "values": {"gain": 1.5},
  "bindings": {"k": 1, "x": 2},
  "graph": {"schema_version": "mathmodel.graph/v1", "nodes": [
    {"id": "k", "kind": "parameter"}, {"id": "x", "kind": "parameter"},
    {"id": "sum", "kind": "add", "inputs": ["k", "x"]}
  ], "outputs": ["sum"]}
```

滑块预览不是最终判决：完整求解、反例重放、约束证书和独立确认仍需走各自流程；拖动频繁时前端应防抖并取消旧请求。
