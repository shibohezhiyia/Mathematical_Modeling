# 配对基准统计

`paired_benchmark_effect` 在相同案例上比较两条流程，记录配对差、预先设定的最小样本门槛和固定种子的 bootstrap 区间。存在同一公式的参数或表示变体时，应传入 `cluster="structure_group"`：函数先在结构组内求平均，再以结构组为独立单位重采样，并分别报告 `sample_count` 与 `independent_unit_count`。样本不足时只给描述性差异；区间不等于零错误风险或显著性证明。
