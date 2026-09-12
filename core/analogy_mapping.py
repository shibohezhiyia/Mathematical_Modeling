"""Conservative cross-domain analogy contracts.

Analogy is useful for proposing a search space, but a domain analogy is not a
causal or physical proof.  This module therefore requires an explicit mapping
of variables, units, interactions, conservation statements and boundaries and
never grants execution authority to an analogy alone.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping


SCHEMA_VERSION = "mathmodel.analogy-mapping/v1"


class AnalogyMappingError(ValueError):
    pass


def _text(value: Any, field: str, limit: int = 500) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise AnalogyMappingError(f"{field}_must_be_nonempty_bounded_text")
    return value.strip()


def validate_analogy_mapping(mapping: Mapping[str, Any], *, confirmed: bool = False) -> dict[str, Any]:
    """Validate a typed analogy and return a reversible search proposal.

    ``confirmed`` only records an explicit user confirmation.  It does not
    turn the proposal into a verified model; downstream unit, data and
    counterexample gates remain mandatory.
    """
    if not isinstance(mapping, Mapping):
        raise AnalogyMappingError("analogy_mapping_must_be_an_object")
    if type(confirmed) is not bool:
        raise AnalogyMappingError("confirmed_must_be_boolean")
    source = _text(mapping.get("source_domain"), "source_domain", 160)
    target = _text(mapping.get("target_domain"), "target_domain", 160)
    variables = mapping.get("variables")
    if not isinstance(variables, list) or not 1 <= len(variables) <= 128:
        raise AnalogyMappingError("variables_must_be_a_bounded_nonempty_list")
    normalized_variables = []
    seen_targets = set()
    for index, item in enumerate(variables):
        if not isinstance(item, Mapping):
            raise AnalogyMappingError(f"variable_mapping_{index}_must_be_an_object")
        source_name = _text(item.get("source"), f"variable_mapping_{index}_source", 128)
        target_name = _text(item.get("target"), f"variable_mapping_{index}_target", 128)
        if target_name in seen_targets:
            raise AnalogyMappingError("target_variables_must_be_unique")
        seen_targets.add(target_name)
        normalized_variables.append({
            "source": source_name, "target": target_name,
            "source_unit": _text(item.get("source_unit"), f"variable_mapping_{index}_source_unit", 80),
            "target_unit": _text(item.get("target_unit"), f"variable_mapping_{index}_target_unit", 80),
            "role": _text(item.get("role", "state"), f"variable_mapping_{index}_role", 80),
        })
    normalized = {"source_domain": source, "target_domain": target, "variables": normalized_variables}
    for field in ("interactions", "conservation_laws", "boundary_conditions"):
        values = mapping.get(field)
        if not isinstance(values, list) or len(values) > 256:
            raise AnalogyMappingError(f"{field}_must_be_a_bounded_list")
        normalized[field] = [_text(value, f"{field}_item", 500) for value in values]
    assumptions = mapping.get("assumptions", [])
    if not isinstance(assumptions, list) or len(assumptions) > 256:
        raise AnalogyMappingError("assumptions_must_be_a_bounded_list")
    normalized["assumptions"] = [_text(value, "assumption", 500) for value in assumptions]
    digest_payload = dict(normalized)
    digest = sha256(json.dumps(digest_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    normalized["confirmed_by_user"] = confirmed
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "confirmed_proposal" if confirmed else "proposal",
        "execution_authorized": False,
        "mapping": normalized,
        "mapping_digest": digest,
        "policy": "analogy_is_a_reversible_search_hint; downstream_type_unit_data_and_counterexample_checks_required",
    }


__all__ = ["SCHEMA_VERSION", "AnalogyMappingError", "validate_analogy_mapping"]
