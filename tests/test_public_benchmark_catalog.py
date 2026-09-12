import json
from pathlib import Path

import pytest

from core.public_benchmark_catalog import PublicBenchmarkCatalog, PublicCatalogError, load_public_catalog


def test_public_catalog_is_tracked_and_contains_official_sources_only():
    path = Path(__file__).resolve().parents[1] / "examples" / "public_benchmark_sources.json"
    catalog = load_public_catalog(path)
    assert len(catalog.sources()) >= 8
    assert {item.provider for item in catalog.sources()} == {"COMAP", "CUMCM"}
    assert all(item.statement_url.startswith("https://") for item in catalog.sources())
    assert catalog.public()["policy"]["contains_problem_text"] is False


def test_catalog_is_deterministic_and_can_filter_by_year_and_track():
    path = Path(__file__).resolve().parents[1] / "examples" / "public_benchmark_sources.json"
    first = load_public_catalog(path)
    second = PublicBenchmarkCatalog.from_payload(json.loads(json.dumps(first.public(), ensure_ascii=False)))
    assert first.digest == second.digest
    assert len(first.sources(year=2025, track="MCM")) == 3


def test_catalog_rejects_non_official_or_solution_material_urls():
    path = Path(__file__).resolve().parents[1] / "examples" / "public_benchmark_sources.json"
    payload = load_public_catalog(path).public()
    bad = dict(payload["sources"][0], source_url="https://example.com/problem.pdf")
    with pytest.raises(PublicCatalogError, match="unapproved_source_host"):
        PublicBenchmarkCatalog.create([bad], name="bad")
    bad = dict(payload["sources"][0], problem="reference_answer")
    with pytest.raises(PublicCatalogError, match="solution_material_forbidden"):
        PublicBenchmarkCatalog.create([bad], name="bad")


def test_catalog_rejects_duplicate_case_ids():
    path = Path(__file__).resolve().parents[1] / "examples" / "public_benchmark_sources.json"
    payload = load_public_catalog(path).public()
    with pytest.raises(PublicCatalogError, match="duplicate_case_id"):
        PublicBenchmarkCatalog.create(payload["sources"][:2] + [
            dict(payload["sources"][2], case_id=payload["sources"][0]["case_id"])
        ], name="bad")


def test_repository_public_comap_catalog_is_valid():
    path = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / "comap_public_catalog.json"
    catalog = load_public_catalog(path)
    assert len(catalog.sources()) == 6
    assert all(item.availability == "source_listed" for item in catalog.sources())
