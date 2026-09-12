# 数据切分与预处理隔离

`audit_split_manifest` 要求训练、验证、最终测试使用不同指纹，并声明预处理统计量只在训练分区拟合。`audit_final_test_isolation` 比较读取最终测试前后的特征选择、预处理拟合、模型拟合和训练行数摘要；最终测试指纹进入这些训练摘要或摘要发生变化时返回 `fail`。这是摘要级审计，不能替代端到端执行追踪。

`audit_point_in_time_features` 逐条检查特征的 `available_at <= prediction_time`，用于滚动特征和时间联表的 point-in-time 审计。
