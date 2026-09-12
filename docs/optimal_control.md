# 有限时域最优控制候选

`solve_linear_quadratic_control` 提供线性离散动力学的有限时域 Riccati 候选。无控制边界时使用标准 LQR 递推；有边界时采用裁剪候选并明确标注，不能把裁剪结果冒充带约束全局最优解。
