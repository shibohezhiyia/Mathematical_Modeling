# 执行准入门

`assess_execution_readiness` 将不同后端必须提供的五类证据统一为一个机器可读门：`type`、`unit`、`source`、`resource` 和 `security`。五项都显式通过才是 `ready`；缺失、`pending` 或 `not_assessed` 会是 `not_assessed`，任何拒绝/失败/不安全状态会是 `blocked`。

候选门 `gate_candidates` 在候选带有 `execution_readiness` 时自动应用该结果。它不会从模型分数、运行成功或字段缺失推断通过，也不会替代具体后端的类型检查、单位推导、来源授权、资源限制或 OS 级沙箱。

```python
from core.execution_readiness import assess_execution_readiness

gate = assess_execution_readiness({
    "type": {"status": "verified", "evidence": ["ir-check-1"]},
    "unit": {"status": "verified", "evidence": ["dimension-check-1"]},
    "source": {"status": "verified", "evidence": ["fact-7"]},
    "resource": {"status": "isolated", "evidence": ["worker-2"]},
    "security": {"status": "verified", "evidence": ["sandbox-2"]},
})
assert gate["status"] == "ready"
```

这是统一证据接口和候选准入协议，不代表全项目所有后端已经接入，也不构成数学正确性证明。
