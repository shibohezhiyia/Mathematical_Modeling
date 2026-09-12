# 有界工作队列

`BoundedWorkQueue` 同时预约待处理数、内存、原生数值线程和 API 调用窗口。超过任一预算会立即返回 `WorkQueueFull`，不会按 CPU 核数无界放开嵌套 BLAS/求解器；任务完成会释放内存和线程预约，API 调用按时间窗口保留。它不是进程级沙箱，进程隔离仍由 `solver_runtime` 负责。
