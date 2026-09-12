# IR genericity guard

`audit_ir_genericity` is a pre-execution heuristic that catches common
competition/domain identifiers accidentally copied into a supposedly generic
IR. It is a guardrail, not a complete ontology or proof of open-world
generality; unknown domain leakage still requires benchmark review.
