from core.module_catalog import audit_module_usage, default_module_catalog, validate_module_catalog


def test_module_catalog_distinguishes_product_and_experimental_entries():
    result = validate_module_catalog(default_module_catalog())
    assert result["status"] == "valid"
    assert result["status_counts"]["active"] >= 2
    assert any(item["status"] == "experimental" for item in result["entries"])
    assert any(item["evidence_scope"] == "proposal_only" for item in result["entries"])


def test_module_catalog_rejects_duplicate_keys():
    catalog = default_module_catalog()
    catalog["entries"].append(dict(catalog["entries"][0]))
    try:
        validate_module_catalog(catalog)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate module key must be rejected")


def test_module_usage_audit_reports_reference_status_without_importing_modules(tmp_path):
    (tmp_path / "core").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "core" / "example.py").write_text("from core.primitive_graph_runtime import execute_primitive_graph\n", encoding="utf-8")
    result = audit_module_usage(tmp_path)
    entry = next(item for item in result["entries"] if item["key"] == "typed_primitive_graph_runtime")
    assert entry["usage_status"] == "referenced"
    assert result["status"] == "audited"
