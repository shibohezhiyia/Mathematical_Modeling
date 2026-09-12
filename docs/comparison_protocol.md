# 对比与消融执行协议

`core.comparison_protocol` 在预注册的 arm × task 网格上运行候选流程，
给每个 arm 相同的任务、预算、随机种子和锁定测试指纹，并把拒绝、超时、
错误和未完成行保留在分母中。`summarize_paired_comparison` 只对两 arm 的
共同完成任务计算配对描述性差异；样本不足时降级，不能把结果称为显著性、
泛化准确率或零错误风险证明。

外部系统（注册表、单次 LLM、AutoML、符号回归或其他 Agent）仍需由调用方
提供合法的 evaluator 和独立未见题数据。本模块不伪造这些系统的结果，也不
允许 evaluator 读取锁定最终测试集训练。

标准 arm 预登记由 `core.standard_comparison_plan` 提供：

```python
from core.standard_comparison_plan import build_standard_comparison_manifest

manifest = build_standard_comparison_manifest(
    task_ids=["case-a", "case-b"],
    fixed_budget={"seconds": 30, "api_calls": 4, "seed": 7},
    final_test_fingerprint="locked-final-v1",
)
```

它固定六个对照臂：注册表基线、单次 LLM、AutoML、符号回归、MM-Agent 风格四阶段和诊断驱动开放组合。这个清单只解决“比较什么、如何同预算比较”，不代表任何一臂已经在真实未见题上胜出。
