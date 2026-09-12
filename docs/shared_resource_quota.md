# 跨进程资源配额

`SharedResourceQuota` 使用 SQLite `BEGIN IMMEDIATE` 建立跨 Python 进程的内存/槽位租约。过期租约会在下一次申请时回收，数据库只保存租约元数据，不保存题面、模型或数值结果。它可以作为多服务部署的共享 admission gate；当前默认求解器仍使用进程内配额，部署方需显式注入共享配额实例，避免把本机路径误当作集群隔离。
