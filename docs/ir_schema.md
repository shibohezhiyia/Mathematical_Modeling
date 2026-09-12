# IR 版本信封

`core.ir_schema` 为中间表示（IR）提供一个与题型无关的传输边界。它解决的是缓存、证据记录和跨阶段传递时的版本与篡改识别问题，不代替数学语义校验。

## 结构

当前版本为 `mathmodel.ir-envelope/v1`，信封固定包含：

```json
{
  "schema_version": "mathmodel.ir-envelope/v1",
  "kind": "primitive_graph",
  "payload": {},
  "payload_digest": "sha256..."
}
```

允许的 `kind` 有 `hypothesis_ir`、`problem_contract`、`search_experiment` 和 `primitive_graph`。摘要使用规范化 JSON（排序键、固定分隔符、禁止非有限数）计算，因此同一 payload 不受字典插入顺序影响。

## 迁移原则

旧的裸 mapping 不会被猜测成某种题型，必须由调用方显式传入 `kind`：

```python
from core.ir_schema import migrate_ir, unwrap_ir

envelope = migrate_ir(legacy_payload, kind="primitive_graph")
payload = unwrap_ir(envelope, expected_kind="primitive_graph")
```

未知字段、摘要不匹配、非 JSON 值、非有限数和超深/超大 payload 会被拒绝。解包后仍必须调用对应的数学 IR/原语验证器；版本信封只保证结构边界和完整性。
