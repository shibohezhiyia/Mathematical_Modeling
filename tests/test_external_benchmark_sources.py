import hashlib
import json
from pathlib import Path

import pytest


def test_external_unseen_manifest_has_real_source_hashes_and_official_urls():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "workspace" / "sealed_benchmarks" / "2026-09" / "public" / "manifest.json"
    source_root = root / "workspace" / "sealed_benchmarks" / "2026-09" / "source"
    if not manifest_path.is_file() or not source_root.is_dir():
        pytest.skip("private sealed benchmark corpus is not checked into public clones")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    assert len(cases) >= 15
    assert all(case["provenance"] == "external_real" and case["split"] == "unseen" for case in cases)
    for case in cases:
        parts = case["id"].split("-")
        contest = parts[2].upper()
        letter = parts[3].upper()
        url = (
            "https://www.contest.comap.com/undergraduate/contests/mcm/contests/"
            f"{parts[1]}/problems/{parts[1]}_{contest}_Problem_{letter}.pdf"
        )
        assert url.startswith("https://www.contest.comap.com/")
        statement = source_root / f"{case['id'].replace('-', '_')}.pdf"
        assert statement.is_file()
        digest = hashlib.sha256(statement.read_bytes()).hexdigest()
        assert digest == case["statement_sha256"]
        assert statement.stat().st_size == case["statement_bytes"]
