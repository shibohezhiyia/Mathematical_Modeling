from hashlib import sha256

import pytest

from core.data_provenance import DataProvenanceError, audit_contamination, build_source_ledger, content_shingle_digests


def _hash(text):
    return sha256(text.encode()).hexdigest()


def test_source_ledger_requires_rights_and_records_content_collisions():
    result = build_source_ledger([
        {"source_id": "a", "uri": "https://example.org/a", "content_sha256": "a" * 64,
         "owner": "owner", "license": "CC-BY", "authorized": True},
        {"source_id": "b", "uri": "file:///tmp/b", "content_sha256": "a" * 64,
         "owner": "owner", "license": "CC-BY", "authorized": False},
    ])
    assert result["status"] == "needs_authorization"
    assert result["duplicate_content_groups"] == [["a", "b"]]


def test_contamination_audit_fails_on_exact_prompt_or_development_overlap():
    shared = _hash("same")
    result = audit_contamination(development_digests=[shared], final_digests=[shared], prompt_digests=[])
    assert result["status"] == "fail"
    with pytest.raises(DataProvenanceError):
        audit_contamination(development_digests=["bad"], final_digests=[])


def test_normalized_shingles_catch_partial_content_overlap_without_retaining_text():
    development = content_shingle_digests("Rainfall drives crop yield in the development sample.")
    final = content_shingle_digests("Rainfall drives crop yield in the final sample.")
    result = audit_contamination(development_digests=[], final_digests=[],
                                 development_shingles=development, final_shingles=final)
    assert result["development_final_shingle_overlap"] > 0
    assert result["status"] == "fail"
    with pytest.raises(DataProvenanceError):
        content_shingle_digests("x", shingle_size=1)


@pytest.mark.parametrize("uri", [
    "http://example.org/data",
    "https://user:secret@example.org/data",
    "file://remote-host/data",
    "https://[broken/data",
])
def test_source_ledger_rejects_unsafe_or_ambiguous_uris(uri):
    with pytest.raises(DataProvenanceError, match="source_uri"):
        build_source_ledger([{
            "source_id": "unsafe", "uri": uri, "content_sha256": "a" * 64,
            "owner": "owner", "license": "CC-BY", "authorized": True,
        }])


def test_source_ledger_rejects_control_characters_and_long_uris():
    base = {"source_id": "unsafe", "content_sha256": "a" * 64,
            "owner": "owner", "license": "CC-BY", "authorized": True}
    with pytest.raises(DataProvenanceError, match="source_uri"):
        build_source_ledger([{**base, "uri": "https://example.org/ok\nnext"}])
    with pytest.raises(DataProvenanceError, match="source_uri"):
        build_source_ledger([{**base, "uri": "https://example.org/" + "x" * 4096}])
