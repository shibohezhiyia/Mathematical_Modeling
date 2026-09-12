# 公开实验配置

`examples/public_experiment_config.json` 固定公开基准运行的题源目录摘要、随机种子、候选/接口/时间/内存预算、方法对照和划分边界。

选模只能使用 `development` 与 `structure_transform`；`unseen` 和 `adversarial` 只能在方案冻结后评价。配置不包含题面、附件或答案，目录摘要变化时配置必须重新生成。

校验：

```powershell
python -m pytest -q tests/test_experiment_config.py tests/test_public_benchmark_catalog.py
```
