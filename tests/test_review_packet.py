import json
from pathlib import Path

from core.blind_benchmark import BlindManifest, build_sealed_case, record_blind_run
from core.review_packet import ReviewPacketError, aggregate_review_forms, build_review_form, build_review_packet


def _fixture(tmp_path: Path):
    statement = tmp_path / "statement.txt"
    statement.write_text("external problem", encoding="utf-8")
    answer = tmp_path / "answer.txt"
    answer.write_text("reference", encoding="utf-8")
    case, _ = build_sealed_case(case_id="case-a", family="optimization", statement_path=statement, answer_path=answer, unlock_token="a" * 32)
    manifest = BlindManifest.create([case], budget={"max_seconds": 10, "max_memory_mb": 32, "max_api_calls": 1, "max_candidates": 1, "seed": 0})
    run = record_blind_run(manifest, case_id="case-a", run_id="run-a", status="completed", budget=manifest.budget, system_version="v1")
    rubric = {"schema_version": "mathmodel.review-rubric/v1", "criteria": ["model", "validation"], "scale": {"min": 0, "max": 4}, "minimum_reviewers": 2}
    return manifest, run, rubric


def test_packet_hides_case_identity_and_forms_are_empty(tmp_path):
    manifest, run, rubric = _fixture(tmp_path)
    result = build_review_packet(manifest, [run], rubric, output_paths={"case-a": "private/result.json"})
    packet = result["packet"]
    assert packet["case_count"] == 1
    assert "case-a" not in json.dumps(packet, ensure_ascii=False)
    assert result["private_mapping"]["mappings"][0]["case_id"] == "case-a"
    form = build_review_form(packet, rubric, reviewer_id="reviewer-1")
    assert form["rows"][0]["scores"] == {"model": None, "validation": None}


def test_packet_rejects_duplicate_runs(tmp_path):
    manifest, run, rubric = _fixture(tmp_path)
    try:
        build_review_packet(manifest, [run, run], rubric)
    except ReviewPacketError as exc:
        assert str(exc) == "duplicate_or_unknown_case"
    else:
        raise AssertionError("duplicate run was accepted")


def test_completed_forms_aggregate_only_with_private_mapping(tmp_path):
    manifest, run, rubric = _fixture(tmp_path)
    result = build_review_packet(manifest, [run], rubric)
    form1 = build_review_form(result["packet"], rubric, reviewer_id="reviewer-1")
    form2 = build_review_form(result["packet"], rubric, reviewer_id="reviewer-2")
    for form in (form1, form2):
        form["rows"][0]["scores"] = {"model": 3, "validation": 2}
    private = result["private_mapping"]
    aggregate = aggregate_review_forms(result["packet"], private, rubric, [form1, form2])
    assert aggregate["status"] == "ready_for_adjudication"
    assert aggregate["rows"][0]["reviewer_count"] == 2
