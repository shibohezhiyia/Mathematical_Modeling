# Research sensitivity API

`POST /api/research/sensitivity` accepts a trusted `base_evaluation` and up to
64 precomputed `variants`. Each variant has `id`, `overrides`, and an optional
`evaluation`. The endpoint only aggregates decision changes; it never executes
callbacks, formulas, or browser-supplied code. Missing variant evaluations are
reported as `not_assessed`.

This keeps slider rendering separate from numerical execution: a trusted model
runner computes each point, then the web layer produces the auditable comparison.
