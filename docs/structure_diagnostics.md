# 结构诊断信号

`diagnose_series_structure` 在昂贵的符号回归、隐状态搜索或仿真前，对一组等时间对齐序列生成三类可复核信号：单调候选、周期候选和可能的变点。每个信号保留分数、窗口或测试滞后数，并返回数据指纹。

```python
from core.structure_diagnostics import diagnose_series_structure

diagnostics = diagnose_series_structure(times, values, ["load", "temperature"])
```

这些统计量只用于改变下一轮的候选搜索空间，例如将周期项加入候选库，不能证明存在周期机制、因果关系或状态切换。当前实现是有界的启发式筛查，不替代正式变点检验、谱显著性检验或独立验证集。
