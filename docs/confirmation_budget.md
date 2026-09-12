# 最终确认预算

`reserve_confirmation_budgets` 在搜索开始前同时预留最终高保真确认和反例搜索预算；预算不足返回 `insufficient_budget`，不会让低保真成绩消费掉最终检查额度。它是预算契约，实际确认仍需调用对应求解器和反例套件。
