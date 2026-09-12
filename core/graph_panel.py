"""Pre-registered, paired finite checks. No heldout-driven winner selection."""
from dataclasses import dataclass, field
import json

from .graph_confirmation import FrozenGraphModel, HeldoutCases
from .graph_experiments import fingerprint
from .model_hypotheses import _canonical, _keys, _require
from .solver_runtime import SolverLimits

PANEL_VERSION = "mathmodel.frozen-model-panel/v1"


@dataclass(frozen=True)
class FrozenModelPanel:
    _json: str = field(repr=False)

    @classmethod
    def from_models(cls, models: dict, *, wall_seconds_per_arm=30, evaluations_per_arm=1000):
        return cls.from_payload({"schema_version": PANEL_VERSION,
            "arms": [{"id": name, "model": model.public()} for name, model in models.items()],
            "wall_seconds_per_arm": wall_seconds_per_arm,
            "evaluations_per_arm": evaluations_per_arm})

    @classmethod
    def from_payload(cls, payload):
        from .confirmation_registry import validate_study_id
        payload = json.loads(_canonical(payload))
        _keys(payload, {"schema_version", "arms", "wall_seconds_per_arm", "evaluations_per_arm"})
        _require(payload["schema_version"] == PANEL_VERSION, "panel_version_mismatch")
        arms = payload["arms"]
        _require(type(arms) is list and 2 <= len(arms) <= 4, "panel_arm_budget")
        wall, evaluations = payload["wall_seconds_per_arm"], payload["evaluations_per_arm"]
        _require(type(wall) in (int, float) and 0.05 <= wall <= 30, "panel_wall_budget")
        _require(type(evaluations) is int and 1 <= evaluations <= 1000, "panel_evaluation_budget")
        names, experiments = set(), set()
        for arm in arms:
            _keys(arm, {"id", "model"})
            validate_study_id(arm["id"])
            _require(arm["id"] not in names, "duplicate_panel_arm")
            names.add(arm["id"])
            model = FrozenGraphModel.from_payload(arm["model"])
            experiments.add(fingerprint(model.public()["experiment"]))
        # Same units, domain, tolerances, development partitions and properties.
        _require(len(experiments) == 1, "panel_experiments_differ")
        return cls(_canonical(payload))

    def public(self):
        return json.loads(self._json)

    @property
    def digest(self):
        return fingerprint(self.public())

    def validate_holdout(self, payload):
        data = None
        for arm in self.public()["arms"]:
            data = HeldoutCases.from_payload(payload, FrozenGraphModel.from_payload(arm["model"]))
        return data


def confirm_frozen_panel(panel, payload, *, registry, study, cancel=None):
    """Consume once for all arms; quotas cannot migrate between arms.

    Caller must freeze the panel before reading holdout data. JSON validation is
    not authentication of this ordering or of external development provenance.
    """
    from .confirmation_registry import _execute_frozen
    panel = FrozenModelPanel.from_payload(panel.public())
    data = panel.validate_holdout(payload)
    attempt = registry.reserve_panel(study, panel, data.public())
    spec = panel.public()
    results = []
    for arm in spec["arms"]:
        # Each worker is supervised separately; sum of quotas excludes parent
        # validation/SQLite overhead. No promise of a hard overall wall deadline.
        limits = SolverLimits(wall_seconds=spec["wall_seconds_per_arm"],
                              max_evaluations=spec["evaluations_per_arm"])
        result = _execute_frozen(FrozenGraphModel.from_payload(arm["model"]), data,
                                 cancel=cancel, limits=limits)
        results.append({"id": arm["id"], "result": result})
    registry.complete(attempt, "panel_completed")
    return {"schema_version": "mathmodel.panel-confirmation/v1",
        "panel_hash": panel.digest, "holdout_hash": data.digest,
        "status": "panel_completed", "arms": results,
        "consumption": registry.inspect(attempt), "may_feed_search": False,
        "winner_selected": False, "all_arms_passed": all(
            a["result"]["status"] == "passed_finite_heldout_checks" for a in results),
        "policy": {"paired_exact_cases": True, "equal_preallocated_worker_quotas": True,
                   "quota_redistribution": False, "statistical_significance_assessed": False,
                   "development_budgets_audited": False, "global_wall_deadline": False,
                   "guard_scope": "same_study_same_registry",
                   "outside_study_freshness_verified": False, "local_store_tamper_proof": False}}
