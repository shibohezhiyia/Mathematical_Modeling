import json

import pytest

from core.modeling_benchmark_suite import build_modeling_benchmark_suite
from core.modeling_confirmation_suite import (
    ModelingConfirmationAlreadyConsumed,
    build_modeling_confirmation_suite,
    complete_modeling_confirmation,
    reserve_modeling_confirmation,
)


def test_confirmation_suite_is_deterministic_separate_and_structure_grouped():
    development = build_modeling_benchmark_suite()
    first = build_modeling_confirmation_suite()
    second = build_modeling_confirmation_suite()
    assert len(first) == 39
    assert len({case.structure_group for case in first}) == 13
    assert not ({case.case_id for case in development} & {case.case_id for case in first})
    assert [case.public_metadata() for case in first] == [case.public_metadata() for case in second]
    assert {case.source_group for case in first} == {"generated-modeling-confirmation-v1"}


def test_confirmation_suite_changes_names_orders_and_numerical_scales():
    development = {case.structure_group: case for case in build_modeling_benchmark_suite()}
    confirmation = build_modeling_confirmation_suite()
    renamed = [case for case in confirmation if case.case_id.endswith("-v2")]
    assert all(all(item["name"].startswith("raw_") for item in case.public_input["attachments"])
               for case in renamed)
    algebra = next(case for case in confirmation
                   if case.structure_group == "modeling-algebra-affine")
    assert algebra.hidden_reference["coefficients"] != development[
        algebra.structure_group
    ].hidden_reference["coefficients"]


def test_confirmation_seed_is_consumed_atomically_even_before_completion(tmp_path):
    marker = reserve_modeling_confirmation(
        tmp_path, protocol_id="modeling-internal-confirmation-v1", seed=7,
    )
    with pytest.raises(ModelingConfirmationAlreadyConsumed, match="already_consumed"):
        reserve_modeling_confirmation(
            tmp_path, protocol_id="modeling-internal-confirmation-v1", seed=7,
        )
    complete_modeling_confirmation(marker, outcome="completed", report_path="report.json")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["report_path"] == "report.json"
