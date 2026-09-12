# OpenModelBench 基准协议

`core/open_model_bench.py` 提供基准清单、结果协议和一个薄的 `BenchmarkRunner` 适配器，不替代真实题目数据。每个案例固定一个划分：

- `development`：允许搜索与调参，但需要记录重复使用。
- `unseen`：冻结方案后的未见题。
- `structure_transform`：变量改名、单位变化、故事替换或结构等价变形。
- `adversarial`：定义域、噪声、机制切换、矛盾条件等压力案例。

案例还必须标注建模族：`data`、`pure_mechanistic`、`multi_table`、`optimization`、`dynamics`。数据通过 SHA-256 指纹引用；同一个数据指纹不能同时出现在开发和未见划分中。

结果记录必须携带 `pass/fail/unresolved/not_run` 状态。失败必须归因到 `representation/proposal/type_check/compile/numeric/evidence/validation` 之一。协议只统计完成率和失败阶段，不在没有独立真值时伪造准确率；真实模型运行、数据读取和最终判定仍由上层编排器负责。

结果可选填 `contract_correct`、`ir_effective`、`executable`、`numerically_correct`、`constraint_violation_rate`、`counterexample_found` 和 `interval_coverage`。汇总只对实际提供的值报告样本数和均值，缺失值不会被当成失败或零分。

示例：

```python
from core.open_model_bench import OpenModelBench

bench = OpenModelBench.create([
    {"id": "dev-001", "split": "development", "family": "data",
     "statement": "研究变量关系。"},
    {"id": "unseen-001", "split": "unseen", "family": "optimization",
     "statement": "在约束下优化目标。"},
])
result = bench.record_result("dev-001", status="fail", failure_stage="numeric")
print(bench.summarize([result]))
```

仓库中的最小公开清单位于 `examples/open_model_bench_minimal.json`，可在不执行任何模型的情况下验证：

```powershell
python scripts/validate_open_model_bench.py examples/open_model_bench_minimal.json
```

命令输出清单摘要和稳定指纹；它不会读取数据文件，也不会调用外部 API。

仓库还提供五类有界合成夹具，用来验证“案例能被加载、分类和审计”，不作为现实准确率：

```powershell
python scripts/validate_synthetic_bench.py
```

该命令只公开每个夹具的家族、数据指纹和检查项；事实数据仍留在进程内，不进入公开基准清单。

多个方法可以用 `compare_runs` 做开发集配对对照。它拒绝 `unseen`/`adversarial` 作为选模输入，保留缺失案例分母，并固定返回 `winner_selected=false`；最终方案选择必须在冻结规则后另行执行。

`BenchmarkRunner.run` 将一个外部 callable 逐案例接入协议：callable 只接收公开案例字典并返回 `status`、可选 `failure_stage`、指标和资源字段。结果格式错误会记为 `validation` 失败，未捕获异常会记为 `proposal` 失败，且不会把堆栈、题面或用户数据写入结果。它只负责编排和归因，不会替 runner 声称数值正确。
