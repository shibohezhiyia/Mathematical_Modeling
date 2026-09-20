from pathlib import Path

import pytest

from core.external_dataset_catalog import ExternalDatasetError, load_catalog, validate_catalog


def test_public_catalog_is_pinned_and_allowlisted():
    path = Path(__file__).parents[1] / "examples" / "external_dataset_catalog.json"
    payload = load_catalog(path)
    assert len(payload["datasets"]) == 6
    assert all(item["sha256"] and item["bytes"] > 0 for item in payload["datasets"])


def test_catalog_rejects_non_https_or_untrusted_host():
    with pytest.raises(ExternalDatasetError, match="dataset_url_not_allowlisted"):
        validate_catalog({"schema_version": "mathmodel.external-dataset-catalog/v1", "datasets": [{
            "id": "x", "title": "x", "provider": "x", "license": "x", "task_family": "x",
            "format": "zip", "url": "http://evil.example/x.zip"
        }]})


def test_catalog_rejects_path_traversal_filename(tmp_path):
    from core.external_dataset_catalog import download_dataset
    row = {"id": "x", "title": "x", "provider": "x", "license": "x", "task_family": "x",
           "format": "zip", "filename": "../x.zip", "url": "https://archive.ics.uci.edu/x.zip"}
    with pytest.raises(ExternalDatasetError, match="dataset_filename_invalid"):
        download_dataset(row, tmp_path)


def test_download_rejects_archive_path_escape(tmp_path, monkeypatch):
    import zipfile
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.csv", "x")
    row = {"id": "x", "title": "x", "provider": "x", "license": "x", "task_family": "x",
           "format": "zip", "filename": "source.zip", "url": "https://archive.ics.uci.edu/x.zip",
           "bytes": archive.stat().st_size, "sha256": __import__("hashlib").sha256(archive.read_bytes()).hexdigest()}
    from core import external_dataset_catalog as module
    class _Response:
        def __enter__(self): return archive.open("rb")
        def __exit__(self, *args): return False
    monkeypatch.setattr(module, "urlopen", lambda *args, **kwargs: _Response())
    with pytest.raises(ExternalDatasetError, match="dataset_archive_path_escape"):
        module.download_dataset(row, tmp_path / "out", extract=True)


def test_download_reuses_verified_existing_file(tmp_path):
    import hashlib
    target = tmp_path / "x.zip"
    target.write_bytes(b"abc")
    row = {"id": "x", "title": "x", "provider": "x", "license": "x", "task_family": "x",
           "format": "zip", "filename": "x.zip", "url": "https://archive.ics.uci.edu/x.zip",
           "bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    from core.external_dataset_catalog import download_dataset
    receipt = download_dataset(row, tmp_path)
    assert receipt["reused"] is True and receipt["verified"] is True
