# Worker 权限与磁盘配额

`core.worker_permissions` 提供 Python audit hook 之上的防御层：路径必须位于显式
允许目录，默认拒绝网络和子进程；`directory_usage_bytes` 统计不跟随符号链接的
常规文件大小，`enforce_write_quota` 返回当前用量和剩余额度。对已达到额度的
新建/追加打开操作会拒绝，任务结束后仍应由宿主或 `solver_runtime` 做最终目录扫描。

这不是操作系统沙箱。Windows Job Object、受限令牌、目录 ACL、容器/命名空间和
网络隔离仍需按部署平台实测；audit hook 也无法知道一次写入的未来大小，不能替代
OS 级隔离。
