import pytest

from core.analogy_mapping import AnalogyMappingError, validate_analogy_mapping


def _mapping():
    return {
        "source_domain": "fluid flow",
        "target_domain": "bike rebalancing",
        "variables": [{"source": "flow", "target": "bike_rate", "source_unit": "bike/hour", "target_unit": "bike/hour", "role": "state"}],
        "interactions": ["inflow minus outflow"],
        "conservation_laws": ["station inventory is conserved between events"],
        "boundary_conditions": ["inventory cannot be negative"],
        "assumptions": ["stations are observed at event times"],
    }


def test_analogy_requires_explicit_mapping_and_never_authorizes_execution():
    proposal = validate_analogy_mapping(_mapping())
    assert proposal["status"] == "proposal"
    assert proposal["execution_authorized"] is False
    confirmed = validate_analogy_mapping(_mapping(), confirmed=True)
    assert confirmed["status"] == "confirmed_proposal"
    assert confirmed["mapping_digest"] == proposal["mapping_digest"]


@pytest.mark.parametrize("field", ["variables", "interactions", "conservation_laws", "boundary_conditions"])
def test_analogy_rejects_missing_or_malformed_evidence(field):
    mapping = _mapping()
    mapping[field] = [] if field == "variables" else "not-a-list"
    with pytest.raises(AnalogyMappingError):
        validate_analogy_mapping(mapping)


def test_analogy_rejects_duplicate_target_variables():
    mapping = _mapping()
    mapping["variables"].append({"source": "pressure", "target": "bike_rate", "source_unit": "Pa", "target_unit": "bike/hour"})
    with pytest.raises(AnalogyMappingError, match="target_variables_must_be_unique"):
        validate_analogy_mapping(mapping)
