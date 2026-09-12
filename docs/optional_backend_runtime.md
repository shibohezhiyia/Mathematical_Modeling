# 可选数学后端运行时

`core.optional_backend_runtime.execute_optional_backend` 为 SciPy、SymPy、
CVXPY、Pyomo 和 JAX 提供显式的结构化入口：SciPy 只能调用已登记的通用
求解契约，SymPy 只能从安全 AST 构造表达式，JAX 只能执行受限可微 IR，
CVXPY 只接受有界线性规划字段。未安装依赖、缺少 Pyomo 外部求解器或后端
运行失败都会返回 `unavailable`/`incomplete`/`error`，不会伪造数学结论。

这解决的是后端接入边界，不代表所有后端都具备相同求解能力，也不替代独立
残差、约束和稳定性复核。
