# 交互、UDE 与 PDE 候选空间

`screen_interactions` 对多变量矩阵做有界相关筛选，仅输出非因果候选边；`build_ude_contract` 描述已知右端加受限神经修正项的类型契约；`fit_ude_correction` 提供无额外深度学习依赖的有界基函数修正、时间留出和扰动稳定性审计；`build_pde_library_contract` 描述有界网格导数候选库。所有结果都不授予因果或物理解释，必须由边界条件、单位、留出残差和稳定性检查继续验证。
