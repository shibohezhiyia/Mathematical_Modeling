# 分层不确定性审计

`build_uncertainty_audit` 将题意、结构、参数、数值四层的有限输出变化与独立留出样本上的区间/概率校准分开计算。没有独立校准数组时返回 `partial`，不会把开发段拟合或分层方差写成置信度。

```python
from core.uncertainty_audit import build_uncertainty_audit

report = build_uncertainty_audit(evaluations, unit_signature="m", interval={
    "actual": actual, "lower": lower, "upper": upper,
})
```

报告中的校准是有限样本经验覆盖或可靠性统计，不是分布无关覆盖证明；四层方差是经验分解，也不是后验方差。

Web 端对应接口为 `POST /api/research/uncertainty-audit`，请求体传入 `evaluations`，可选 `interval` 或 `probability`。接口不会复用开发输出作为校准样本；缺少独立校准数据时返回 `partial`。
