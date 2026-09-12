# 当前系统与旧版基线对照

`core.system_benchmark.compare_frozen_systems` 在同一训练/测试切分上运行：

1. 冻结的 Ridge + 中位数填补 + 标准化基线；
2. 当前 `ModelingEngine`，包含预处理、交叉验证、结构化模型组合和候选选择。

两者只共享训练特征和训练标签，测试标签仅交给独立评分函数。结果保留
`baseline`、`current_system`、最佳模型、运行耗时和预测摘要，失败返回
`not_assessed`，不会把失败当成低分。

## 公开数据实跑

运行：

```powershell
$env:PYTHONPATH='.'
python scripts/run_external_data_benchmark.py --include-current-system `
  --output data/reports/external_data_system_comparison.json
```

2026-09-11 的 6 个 UCI 案例全部完成。当前系统相对 Ridge 的任务级平均
RMSE 变化为 **-0.237916**；逐任务效果为：

```text
[-0.163431, 0.009186, -0.613776, -0.214250, 0.176920, -0.622147]
```

这证明“当前系统确实进入了同一冻结切分的数值对照”，但仍是公开标签的
描述性证据，不是数学建模题准确率、专家评分或统计显著性证明。最终结论
仍需封存题目和独立评审。

本次运行对应的冻结摘要为
`d87704112e75060d05f9b517d494debac8faf99561579c66c3316a9e026c287d`，机器可读
记录见 `examples/system_comparison_20260911.json`。

## 统一动态编译入口

`POST /api/research/dynamic-compile` 接受 `primitive_graph`、`ode_cegis`、
`optimization_cegis` 和 `external_method` 四类类型化 JSON 合同，禁止
`source/code/python/command/path/module/import` 字段。它只是受限后端路由，
不允许模型提交 Python 源码，也不把候选执行结果自动升级为现实正确性。

## 当前仍需外部条件的项目

- 封存数学题、双人盲评和 20+ 独立题的功效/显著性实验；
- 非规则网格 PDE、任意控制/刚性混合系统、完整动态图因果发现；
- 由部署方提供并保护的 Windows/Linux OS 隔离证明和实机攻击测试；
- 四层不确定性的独立真值校准。当前新增的
  `assess_four_layer_uncertainty` 是条件情景方差分解，不是后验概率。
