# 离散外微分契约

`build_simplicial_complex` 为顶点、带方向边和三角面构造有限复形，生成 `d0`/`d1` incidence 矩阵并验证 `d1·d0=0`。`apply_exterior_derivative` 只执行已验证形状的有限形式映射。

这是一层可供 PDE/Decapodes 风格后端使用的拓扑与形式次数契约，不是完整范畴论编译器，也不由拓扑合法性推出物理正确性。
