# 模型族 CEGIS 适配契约

`core.model_family_adapters` 把每个模型族的 `compile → evaluate → diagnose →
patch → replay` 接口接入统一的有界 CEGIS 控制器。编译或资源异常只产生
`not_assessed`，不会被当成数学反例；只有评价器明确返回失败和见证时才会进入
反例档案。当前模块提供通用协议和回归测试，不意味着 ODE、优化、PDE 或多表
已经全部实现了具体适配器。发布前仍需为每个模型族提供独立求解、反例重放和
冻结确认证据。
