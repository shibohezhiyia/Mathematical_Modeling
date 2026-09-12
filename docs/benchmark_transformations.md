# 基准结构变换

`core.benchmark_transformations` 提供不执行代码的变体生成器，用于验证系统是否只记住题面词汇。它支持边界明确的变量改名、单位词替换和叙事词替换，并把结果标为 `structure_transform`。

变体默认清除数据指纹和真值标记。若要保留真值，调用方必须先生成或审计独立数据，再显式传入新的数据指纹；不能把开发集的指纹复制到变体中。这个约束避免“故事变了、数据没变”被误计为结构泛化。

```python
from core.benchmark_transformations import transform_case

variant = transform_case(
    development_case,
    transform_id="rename_symbols",
    symbol_map={"x": "state"},
    story_map={"设备": "储能单元"},
    unit_map={"m/s": "km/h"},
)
```

单位字符串替换不会自动换算数值；如果单位改变会影响数据，必须生成独立数据并传入新指纹。该协议不代表模型已经通过结构变换题；变体仍需进入独立运行、反例检查和最终留出评价。
