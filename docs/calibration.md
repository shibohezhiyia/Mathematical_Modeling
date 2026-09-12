# 预测校准检查

`core.calibration` 将预测质量和模型拟合分开审计：

- `assess_interval_calibration` 检查真实值落在预测区间的经验覆盖率、上下界越界率、平均/中位区间宽度和 interval score。
- `assess_probability_calibration` 检查二分类概率的 Brier 分数，并输出有限分箱的可靠性表和 ECE。

这些函数应接收没有参与调参的确认样本。样本少于 10 条时仍返回描述性统计，但状态为 `not_assessed`；需要正式置信区间、分布漂移校准或后验推断时必须另建概率模型。

```python
from core import assess_interval_calibration

report = assess_interval_calibration(
    actual=[1, 2, 3], lower=[0, 1, 2], upper=[2, 3, 4],
    nominal_coverage=0.9,
)
```

系统不会把覆盖率一次达标写成“模型正确”，也不会将 Brier/ECE 解释为因果效应或现实世界保证。
