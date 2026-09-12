# 受限 SymPy 后端

`compile_sympy_expression` 先使用项目的 `SafeNumericExpression` 校验表达式，再从 AST 节点逐一构造 SymPy 对象，不调用 `sympify`、`eval` 或属性访问。SymPy 结果是符号候选，不等于数值求解或数学证明，仍需量纲、域、资源和数值证据门。
