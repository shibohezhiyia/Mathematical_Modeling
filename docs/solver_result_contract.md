# Solver result contract

`normalize_solver_result` provides one evidence envelope for heterogeneous
backends. It standardizes status, value, residual, error-bound status,
iteration count, and resource usage. Missing error bounds stay explicitly
`not_assessed`; normalization never turns a solver status into a proof.
