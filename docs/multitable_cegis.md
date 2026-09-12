# 多表结构 CEGIS

`core.multitable_cegis` 把跨表建模中最容易出错的部分显式化：表粒度、复合
键、连接方向和一对多聚合。右表先按键聚合，再以 `many_to_one` 合并，避免
静默产生笛卡尔积。候选失败时只允许在有限聚合集合
`mean/sum/first/count` 中做一次变异；键不存在、连接不连通或超过行预算都
保留为结构失败/未决，而不是猜测列对应关系。

该模块可以通过统一入口 `POST /api/research/dynamic-compile` 的
`kind=multitable_cegis` 调用，也可直接调用 `run_multitable_cegis`。它验证的是
数据结构和可执行性，不是因果关系或现实机制证明。
