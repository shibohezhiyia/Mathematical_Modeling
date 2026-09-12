# 结论证据证书

`build_conclusion_certificate` 强制报告同时携带适用边界、假设、残差、约束检查和反例记录。没有反例只会得到 `tested_not_falsified`，不会得到全局正确性或数学证明；存在反例则标记为 `counterexample_found`。
