# 可逆尺度变换

`fit_reversible_scaler` 为数值条件差异较大的变量生成带单位签名的标准化或 min-max 变换。变换保留中心、尺度、常量列和摘要指纹；`inverse_transform` 可恢复原始单位，`audit_roundtrip` 检查回变换误差。

尺度变换不会把不同单位变成可互换量，也不会替代量纲检查、约束检查或模型验证。
