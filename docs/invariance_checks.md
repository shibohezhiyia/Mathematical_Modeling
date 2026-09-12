# Metamorphic/invariance checks

`run_invariance_checks` provides a bounded protocol for unit round trips,
entity reorderings, equivalent input representations, or mesh/step changes.
The caller supplies the transformation and comparison rule, so the system
does not invent which quantities should be invariant. A finite pass is reported
as `tested_not_falsified`; it is never promoted to a global mathematical proof.
