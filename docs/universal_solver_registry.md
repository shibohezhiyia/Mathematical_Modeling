# 通用求解器能力清单

`UniversalSolverRegistry` 现在同时承担两件事：按版本执行已经验证的数学契约，以及提供不执行用户数据的能力描述。能力描述的 schema 是 `mathmodel.solver-capability/v1`，包含数学族、输入契约、复杂度估计和后端映射，供结构编译器在执行前做路线选择。

```python
from core.universal_math_solvers import UniversalSolverRegistry

registry = UniversalSolverRegistry()
print(registry.describe("maximum_flow/v1"))
all_backends = registry.describe()
```

描述不是正确性证明，也不替代 `UniversalRelationValidator`。复杂度字段是保守的规划提示；实际运行仍受输入规模、数值条件、求解器状态和资源监督约束。未知或自定义执行器若未提供能力元数据会被标为 `custom/unknown`，不会因此获得更高可信等级。

扩展执行器时可以一并提供能力元数据：

```python
registry.register(
    "my_relation/v1", execute,
    capability={
        "family": "algebra",
        "input_contract": "finite scalar mapping",
        "complexity": "O(1)",
        "backend": "my-safe-backend",
    },
)
```

执行器仍必须接收结构化契约并自行完成有限性、形状、单位和结果复核；能力清单禁止把任意 Python 对象或函数传入数值环境。
