# 规则网格 PDE 候选发现

`core.pde_discovery.discover_1d_pde` 在规则的一维空间网格和有序时间网格上拟合有限候选；`discover_2d_pde` 进一步支持规则二维空间网格：

\[
u_t = a u + b u_x + c u_{xx}.
\]

它使用训练时间段拟合系数，在末段时间留出上计算 PDE 残差，并单独计算空间边界点残差。超过容差会返回 `candidate_rejected_by_validation` 和有限反例，不会把拟合方程写成 PDE 定理。

二维执行器限制在 `u, u_x, u_y, u_xx, u_yy`，使用时间留出、显式 Euler 滚动和步长为 2 的粗网格复核；三维执行器进一步支持 `u_z/u_zz`，同样报告 CFL 风险。它们仍不是任意 PDE、混合导数、不规则网格或边界物理的通用求解器。`pde_find` 没有提供时空场时仍返回旧的类型化特征库规划，提供 `times`、`coordinates`、`field`、二维的 `times`、`x_coordinates`、`y_coordinates`、`field` 或三维再加 `z_coordinates` 才会执行候选拟合。

可选契约字段 `field_dimensions`、`coordinate_dimensions`、`noise_scale`、`boundary_values` 和 `field_mask` 会被校验并写入证据。一维和二维规则网格都支持逐时 Dirichlet 边界：声明的边界会在前向滚动的每一步覆盖对应边缘，并以验证时间段的 `boundary_condition_rmse` 复核；二维边界还会检查四角冲突。未声明边界时仍只固定初始边缘，结果标记为 `initial_boundary_only`。规则有限差分暂不跨缺失值插值；掩码中出现缺失会明确拒绝，避免把插值假设伪装成观测事实。
