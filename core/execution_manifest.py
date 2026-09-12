"""Content-addressed execution metadata for reproducible solver runs."""
from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any, Mapping


class ExecutionManifestError(ValueError):
    pass


def _manifest_key(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise ExecutionManifestError("manifest_must_be_mapping")
    required = ("schema_version", "input_hash", "random_seed", "dependencies", "code_sha256", "log_digest")
    if any(key not in manifest for key in required):
        raise ExecutionManifestError("manifest_fields_missing")
    if manifest.get("schema_version") != "mathmodel.execution-manifest/v1":
        raise ExecutionManifestError("manifest_schema_mismatch")
    dependencies = manifest.get("dependencies")
    if not isinstance(dependencies, Mapping):
        raise ExecutionManifestError("manifest_dependencies_invalid")
    return {"schema_version": manifest["schema_version"], "input_hash": manifest["input_hash"],
            "random_seed": manifest["random_seed"], "dependencies": dict(sorted(dependencies.items())),
            "code_sha256": manifest["code_sha256"], "log_digest": manifest["log_digest"],
            "metadata": manifest.get("metadata", {})}


def compare_execution_manifests(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Compare execution conditions without claiming output equivalence.

    This is a reproducibility *precondition* check: a matching key means the
    runs used the same declared input, seed, dependency versions and generated
    code digest.  Numerical outputs still need an independent tolerance check.
    """
    left_key, right_key = _manifest_key(left), _manifest_key(right)
    differences = sorted(key for key in left_key if left_key[key] != right_key[key])
    return {"schema_version": "mathmodel.execution-manifest-compare/v1",
            "compatible": not differences, "differences": differences,
            "same_declared_execution": not differences,
            "policy": "matching_manifest_is_not_output_equivalence_proof"}


def build_execution_manifest(
    *, run_id: str, input_hash: str, random_seed: int | None,
    dependencies: Mapping[str, str], code: str | None = None,
    log_digest: str | None = None, metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if type(run_id) is not str or not run_id or len(run_id) > 128:
        raise ExecutionManifestError("invalid_run_id")
    if type(input_hash) is not str or len(input_hash) != 64 or any(ch not in "0123456789abcdef" for ch in input_hash):
        raise ExecutionManifestError("invalid_input_hash")
    if random_seed is not None and (type(random_seed) is not int or not 0 <= random_seed <= 2**63 - 1):
        raise ExecutionManifestError("invalid_random_seed")
    if not isinstance(dependencies, Mapping) or not dependencies or len(dependencies) > 256:
        raise ExecutionManifestError("invalid_dependencies")
    normalized_deps = {}
    for name, version in dependencies.items():
        if type(name) is not str or not 1 <= len(name) <= 128 or type(version) is not str or not 1 <= len(version) <= 128:
            raise ExecutionManifestError("invalid_dependencies")
        normalized_deps[name] = version
    if code is not None and (type(code) is not str or len(code.encode("utf-8")) > 2_000_000):
        raise ExecutionManifestError("invalid_generated_code")
    if log_digest is not None and (type(log_digest) is not str or len(log_digest) != 64):
        raise ExecutionManifestError("invalid_log_digest")
    payload = {"schema_version": "mathmodel.execution-manifest/v1", "run_id": run_id,
               "input_hash": input_hash, "random_seed": random_seed,
               "dependencies": dict(sorted(normalized_deps.items())),
               "code_sha256": None if code is None else sha256(code.encode("utf-8")).hexdigest(),
               "log_digest": log_digest, "metadata": dict(metadata or {})}
    if len(payload["metadata"]) > 64:
        raise ExecutionManifestError("metadata_too_large")
    payload["manifest_digest"] = sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
    payload["policy"] = "metadata_only_no_raw_code_or_logs_required"
    return payload


__all__ = ["ExecutionManifestError", "build_execution_manifest", "compare_execution_manifests"]
