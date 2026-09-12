# 持久中间量缓存

`PersistentCompileCache` 把 `IncrementalCompileCache` 的中间编译产物以 JSON 原子替换方式保存到指定目录。键仍包含编译器版本、来源快照、域契约、节点参数和上游键；预测、判决、证明、代码、traceback 和 pickle 永远不进入文件。

缓存文件损坏或版本不符时安全退回冷编译。`clear()` 只删除当前目录下的 `intermediate_cache.json`，便于清理；命中缓存也不跳过独立数值验证。
