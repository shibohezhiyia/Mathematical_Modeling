# 诊断到搜索提示

`core.diagnostic_router` 把低成本诊断结果转换成可追踪、可撤销的结构搜索提示：

- `monotone_candidate` → 单调约束、饱和响应；
- `periodic_candidate` → 正余弦基底、振荡状态、季节滞后；
- `change_point_candidate` → 分段机制、事件切换、阈值约束；
- 选参段残差记忆 → 滞后、外部输入、数学潜状态；
- 误差尺度模式 → 异方差、乘性噪声、稳健损失。
- 预测模型的跨折不稳定、残差结构/偏差、异方差候选和分类错误率 → 稳定性、校准、非线性和标签核验方向；
- 聚类可信度警告与有序时序结构信号 → 距离/分组、异常复核、周期、变点和单调性方向。

提示只包含候选原语 ID、来源证据、优先级和澄清问题。开发、编译前和执行审计可以进入软提示；最终测试阶段的诊断不会进入路由反馈。它们不是硬约束，不能直接修改 IR，不能证明因果边、隐藏状态或真实机制。搜索器使用提示后必须重新通过类型、量纲、定义域、资源和留出验证。

典型调用：

```python
from core import route_structure_diagnostics

hints = route_structure_diagnostics(structure_diagnostics)
for hint in hints["hints"]:
    print(hint["family"], hint["candidate_primitives"])
```

这一步解决的是“诊断结果不会反过来改变搜索空间”的工程断点；它还没有声称某个提示能提高未见题成功率，必须在固定预算基准上单独评估。
