# 通用原语运行时

`core.primitive_runtime.PrimitiveRuntime` 是无题目分支的受限数值入口。它只接受结构化 payload，不执行公式字符串、回调或外部模块。

当前可用的几何与事件原语包括：

- `distance`：欧氏、曼哈顿和 haversine 距离；
- `segment_intersection`：二维线段相交和交点；
- `region_membership`：闭盒或多边形成员判定，边界计入；
- `line_of_sight`：多边形障碍物遮挡判定；
- `interval_union`：有界区间并集和总测度。

统计原语包括 `normal_log_likelihood`、固定种子 `bootstrap_mean` 和 `permutation_test`。每个操作都有有限性、形状、规模和随机预算检查；bootstrap/置换结果是经验样本证据，不是总体证明。

同样的契约已注册到 `UniversalRelationValidator`、`UniversalSolverRegistry` 和受监督 worker。执行结果仍需沿用残差、边界、稳定性和反例证据门。
