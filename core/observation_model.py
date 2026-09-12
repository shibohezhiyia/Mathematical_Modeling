"""Typed observation-model contracts for mechanism/measurement competition."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence


class ObservationModelError(ValueError):
    """Raised for incomplete or contradictory observation contracts."""


@dataclass(frozen=True)
class ObservationModelSpec:
    observed: tuple[str, ...]
    latent: tuple[str, ...]
    relation: str
    delay: float = 0.0
    bias_terms: tuple[str, ...] = ()
    noise: str = "unspecified"
    grouping: tuple[str, ...] = ()
    missingness: str = "unspecified"

    def __post_init__(self) -> None:
        for name, values in (("observed", self.observed), ("latent", self.latent),
                             ("bias_terms", self.bias_terms), ("grouping", self.grouping)):
            if not isinstance(values, tuple) or any(not isinstance(item, str) or not item.strip() for item in values):
                raise ObservationModelError(f"invalid_{name}")
            if len(set(values)) != len(values):
                raise ObservationModelError(f"duplicate_{name}")
        if not self.observed or not isinstance(self.relation, str) or not self.relation.strip():
            raise ObservationModelError("observation_relation_required")
        if not isinstance(self.delay, (int, float)) or self.delay < 0:
            raise ObservationModelError("delay_must_be_nonnegative")
        if self.noise not in {"unspecified", "gaussian", "poisson", "heteroscedastic", "robust", "empirical"}:
            raise ObservationModelError("unsupported_noise_model")
        if self.missingness not in {"unspecified", "none", "MCAR", "MAR", "MNAR", "structural"}:
            raise ObservationModelError("unsupported_missingness_model")

    def public(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["observed"] = list(self.observed)
        payload["latent"] = list(self.latent)
        payload["bias_terms"] = list(self.bias_terms)
        payload["grouping"] = list(self.grouping)
        payload["schema_version"] = "mathmodel.observation-model/v1"
        payload["status"] = "typed_contract_not_empirical_proof"
        return payload


def validate_observation_competition(specs: Sequence[ObservationModelSpec]) -> dict[str, Any]:
    if not isinstance(specs, Sequence) or isinstance(specs, (str, bytes)) or not specs:
        raise ObservationModelError("observation_specs_required")
    if any(not isinstance(spec, ObservationModelSpec) for spec in specs):
        raise ObservationModelError("observation_specs_must_be_typed")
    return {
        "schema_version": "mathmodel.observation-model-set/v1",
        "count": len(specs),
        "specs": [spec.public() for spec in specs],
        "dimensions": sorted({name for spec in specs for name in spec.observed}),
        "policy": "process_and_observation_models_remain_competing_hypotheses",
    }


__all__ = ["ObservationModelError", "ObservationModelSpec", "validate_observation_competition"]
