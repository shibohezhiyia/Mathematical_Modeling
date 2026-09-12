# Safe diagnostics

`core.log_safety.safe_diagnostic` is the boundary between private execution
logs and user/model feedback. It keeps stable error codes and minimal witness
metadata, removes traceback/API-key/raw-path fields through the existing
redaction policy, and enforces a byte budget. Internal logs may remain richer
under local access controls; this function must be used for persisted evidence
and LLM prompts.
