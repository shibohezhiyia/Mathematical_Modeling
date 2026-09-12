# Same-budget paired benchmark

`run_paired_budget_benchmark` evaluates baseline and treatment on the same
cases, repeats and deterministic seeds. It records latency P50/P95, validity,
API calls, cost, rejected/error rows and the full paired rows. Failed or
unfinished runs remain in the denominator. The report is descriptive; a
statistical significance claim requires a pre-registered test and independent
task set.

`core.acceptance_risk_gate.assess_repair_acceptance_risk` 可在此结果之上应用预登记风险门：样本不足返回 `not_assessed`；样本足够时，只有配对有效率提升的 bootstrap 下界达到阈值且治疗组错误通过率的 Wilson 上界低于阈值，才返回 `pass`。这仍然不是未见题泛化证明。
