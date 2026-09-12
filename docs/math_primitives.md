# 最小数学原语签名

`core/math_primitives.py` 是开放世界编译器的接口层。它把候选模型表示为带类型、量纲和待验证义务的节点，而不是让模型直接输出 Python 或字符串公式。

当前初版覆盖：变量、坐标、常量、方程、微分、积分、分布、事件、图、目标、约束、控制和观测；另提供非执行型 `unknown_mechanism`、`latent_state`、`regime_switch` 节点，用来保留尚未确定的机制、潜在状态和切换结构。每个节点至少声明：

- 输入数量和输入引用；
- 输入/输出语义类型；
- 量纲是否已绑定；
- 需要补充的初值、边界、支持域、控制边界等义务；
- 是否属于当前受控执行器支持的原语。

```python
from core.math_primitives import validate_primitive_node

node = {
    "id": "rate",
    "op": "derivative",
    "inputs": ["state", "time"],
    "kind": "quantity",
    "dimensions": {"L": 1, "T": -1},
    "attributes": {},
}
print(validate_primitive_node(node))
```

组合多个节点时使用 `core.primitive_graph.validate_primitive_graph`。它会检查节点引用、重复 ID、拓扑循环、关键前置义务和等式两端量纲，并返回边数、拓扑深度和保守成本单位供后端做规模路由；返回值只表示 `type_checked_not_executed`，后端仍需执行、记录残差并进行反例测试。

通过签名检查只表示结构类型合法，不表示方程可解、参数可辨识、数值稳定或符合现实。未知领域结构仍应作为待发现机制进入候选搜索，不能通过增加任意 Python 函数绕过这一层。

未决节点不会被当作可执行公式：`unknown_mechanism` 必须声明搜索预算或候选语言，`latent_state` 必须声明识别策略或代理/先验，`regime_switch` 必须声明事件边界语义。只有在后续结构搜索、契约绑定和独立验证完成后，才可映射到具体求解器。
