import pytest

from core.progressive_result import ProgressiveResultError, ProgressiveRun


def test_progressive_result_cannot_publish_preview_as_final():
    run = ProgressiveRun("r1", {"seconds": 5})
    run.add_candidate("m1")
    run.select("m1")
    run.advance("preview", evidence_state="not_assessed")
    with pytest.raises(ProgressiveResultError, match="final_requires"):
        run.advance("final", evidence_state="not_assessed")
    run.advance("confirmation", evidence_state="approved")
    run.advance("final", evidence_state="approved")
    assert run.public()["is_final_claim"] is True


def test_cancelled_run_exposes_failure_stage():
    run = ProgressiveRun("r2", {})
    run.cancel("resource")
    assert run.public()["execution_state"] == "cancelled"
    with pytest.raises(ProgressiveResultError):
        run.advance("preview")


def test_candidate_ids_are_canonical_and_final_can_reuse_approved_state():
    run = ProgressiveRun("r3", {})
    run.add_candidate("  m1  ")
    run.add_candidate("m1")
    run.select(" m1 ")
    run.advance("confirmation", evidence_state="approved")
    run.advance("final")
    assert run.public()["candidate_ids"] == ["m1"]
    assert run.public()["selected_candidate"] == "m1"


def test_cancel_requires_a_nonempty_stage():
    run = ProgressiveRun("r4", {})
    with pytest.raises(ProgressiveResultError, match="failure_stage_required"):
        run.cancel(" ")
