# Guarded equivalence rewrites

`core.equivalence_graph.EquivalenceGraph` is a deliberately small rewrite
layer for the typed mathematical IR. Every edge stores source/target hashes,
the rule, required conditions, and whether the transformed form is executable.

The implementation refuses to erase important mathematics: `x/x -> 1` needs a
nonzero-domain confirmation; implicit-to-explicit form needs a unique-solution
condition; derivative/integral conversion retains the initial/boundary and
integration-constant obligations. A condition-free edge is a proposal, not a
proof or a numerical result.

`rewrite(..., source_hash=...)` can continue from an explicitly recorded form,
so chained transformations retain their source/target hashes. `summary()`
reports verified equivalence classes built only from executable edges; edges
with unresolved conditions stay visible but never collapse candidates for
deduplication. This is an algebraic provenance aid, not a proof assistant.

表达式解析还限制递归深度、元数据长度和有限数值；常量会归一化为有限浮点值，摘要使用 `allow_nan=False`。因此畸形/恶意 JSON 会在重写前拒绝，而不是进入哈希图或消耗后端资源。
