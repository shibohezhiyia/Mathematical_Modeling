# 增量编译缓存

`IncrementalCompileCache` 用于小型数学计算图的增量编译。节点的缓存键包含节点定义、上游节点键、编译器版本、数据源快照和适用域签名；因此修改一个节点会自动重新编译它和所有后继节点，而不影响未变的前驱子图。

`compile_with_consistency(...)` 可在需要时执行一次冷缓存/当前缓存对照。它只比较中间产物、节点键和显式来源依赖，忽略命中统计；返回 `tested_not_falsified` 或 `counterexample`，且始终标记 `verdicts_cached=False`、`validation_skipped=False`。命中缓存不能跳过类型、来源、单位或数值验证。该对照会额外执行一次编译，适合开发/发布验收，不应在每个在线请求中无条件启用。

```python
cache = IncrementalCompileCache(max_entries=256)
result = cache.compile(
    nodes,
    compile_node,
    compiler_version="graph-2",
    source_signature="run-input-sha256",
    domain_signature="SI:v1",
)
```

它只保存中间产物，不保存最终判决、置信度或证据等级；缓存命中也不会跳过独立验证。默认节点 id 是身份的一部分，只有调用方显式提供 `equivalence_key` 才允许跨 id 去重。缓存有条目数和估算字节上限，`clear()` 可安全清理。

对版本化题意，可传入 `source_facts=contract_source_facts(contract)`，并在节点上声明 `source_fact_ids`。这样新增一个用户确认事实时，只要旧节点依赖的引文没有变化，旧中间产物仍可复用；依赖变化的节点及其后继节点会自然失效。没有显式声明来源的节点仍使用全局 `source_signature`，在契约改变时保守重编译。返回的 `node_source_dependencies` 仅是增量编译审计，不表示最终数学证据仍然有效；数值验证、约束检查和留出审计必须重新执行。
