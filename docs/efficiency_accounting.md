# 搜索节省量审计

`build_efficiency_account` 为静态拒绝、低保真淘汰和预算未运行建立显式分母，并要求每个预登记案例都有独立检查记录。稀有事件与慢收敛案例只能通过显式 ID 登记，缺少检查时不能生成可发布的速度报告。

返回的 `saved_evaluation_count` 和 `independent_check_rate` 是描述性遥测，不是准确率或质量提升证明。只有 `eligible_for_speed_report=true` 才允许把节省量与同预算基线并列；它仍不能替代未见题统计基准。
