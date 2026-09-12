import copy

import pytest

from core.proof_certificate import ProofCertificateError, issue_proof_certificate, verify_proof_certificate


def _certificate(status="pass"):
    return issue_proof_certificate(
        statement="x^2 >= 0", assumptions=["x is real"],
        proof_steps=["square of a real number is nonnegative"],
        checker_name="fixture-checker", checker_version="1",
        checker_status=status, scope={"domain": "R"},
    )


def test_certificate_separates_deductive_status_from_checker_failure_and_integrity():
    cert = _certificate()
    assert cert["status"] == "deductively_verified"
    assert verify_proof_certificate(cert)["integrity"] == "pass"
    changed = copy.deepcopy(cert)
    changed["statement"] = "x^2 > 0"
    assert verify_proof_certificate(changed)["integrity"] == "fail"
    assert _certificate("not_assessed")["status"] == "not_assessed"


def test_certificate_rejects_empty_conditions_and_duplicate_steps():
    with pytest.raises(ProofCertificateError, match="assumptions"):
        issue_proof_certificate(statement="x", assumptions=[], proof_steps=["step"],
                                checker_name="c", checker_version="1", checker_status="pass")
    with pytest.raises(ProofCertificateError, match="duplicate_proof_steps"):
        issue_proof_certificate(statement="x", assumptions=["a"], proof_steps=["s", "s"],
                                checker_name="c", checker_version="1", checker_status="pass")
