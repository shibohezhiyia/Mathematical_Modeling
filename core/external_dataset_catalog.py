"""Reproducible intake for small, public external modelling datasets.

The catalog is metadata-only in git. Data files are downloaded into the
ignored ``data/external`` directory, checked against pinned byte size and
SHA-256, and never treated as an independent gold rubric.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse
from urllib.request import Request, urlopen

SCHEMA = "mathmodel.external-dataset-catalog/v1"
_ALLOWED_HOSTS = {"archive.ics.uci.edu", "www.openml.org", "openml.org"}


class ExternalDatasetError(ValueError):
    pass


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ExternalDatasetError(code)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_catalog(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(payload, Mapping) and payload.get("schema_version") == SCHEMA,
             "catalog_schema_mismatch")
    datasets = payload.get("datasets")
    _require(isinstance(datasets, list) and datasets, "catalog_datasets_required")
    seen: set[str] = set()
    clean: list[dict[str, Any]] = []
    for item in datasets:
        _require(isinstance(item, Mapping), "dataset_record_invalid")
        record = dict(item)
        ident = record.get("id")
        _require(isinstance(ident, str) and ident and ident not in seen, "dataset_id_invalid")
        seen.add(ident)
        parsed = urlparse(record.get("url") or "")
        _require(parsed.scheme == "https" and parsed.hostname in _ALLOWED_HOSTS,
                 "dataset_url_not_allowlisted")
        for key in ("title", "provider", "license", "task_family", "format"):
            _require(isinstance(record.get(key), str) and record[key], f"dataset_{key}_required")
        expected = record.get("sha256")
        if expected is not None:
            _require(isinstance(expected, str) and len(expected) == 64 and
                     all(ch in "0123456789abcdef" for ch in expected.lower()),
                     "dataset_sha256_invalid")
        if "bytes" in record:
            _require(type(record["bytes"]) is int and record["bytes"] > 0, "dataset_bytes_invalid")
        clean.append(record)
    return {"schema_version": SCHEMA, "catalog_version": payload.get("catalog_version", 1),
            "datasets": clean}


def load_catalog(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    return validate_catalog(json.loads(source.read_text(encoding="utf-8")))


def dataset(catalog: Mapping[str, Any], dataset_id: str) -> dict[str, Any]:
    for item in catalog.get("datasets", []):
        if item.get("id") == dataset_id:
            return dict(item)
    raise ExternalDatasetError("dataset_not_found")


def _safe_extract_zip(archive_path: Path, extract_root: Path, max_bytes: int) -> list[str]:
    extracted: list[str] = []
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            name = Path(info.filename)
            mode = (info.external_attr >> 16) & 0o170000
            _require(not name.is_absolute() and mode != 0o120000,
                     "dataset_archive_link_or_absolute_path")
            destination = (extract_root / name).resolve()
            _require(destination == extract_root or extract_root in destination.parents,
                     "dataset_archive_path_escape")
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as sink:
                sink.write(source.read(max_bytes + 1))
            _require(destination.stat().st_size <= max_bytes, "dataset_member_size_limit")
            extracted.append(str(destination))
    return extracted


def download_dataset(record: Mapping[str, Any], output_root: str | Path, *, timeout: int = 120,
                     max_bytes: int = 64 * 1024 * 1024, extract: bool = False) -> dict[str, Any]:
    """Download one record atomically and return a verified receipt."""
    clean = validate_catalog({"schema_version": SCHEMA, "datasets": [record]})["datasets"][0]
    name = clean.get("filename") or (clean["id"] + "." + clean["format"].lower().lstrip("."))
    _require(Path(name).name == name and name not in (".", ".."), "dataset_filename_invalid")
    target_dir = Path(output_root).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = (target_dir / name).resolve()
    _require(target.parent == target_dir, "dataset_output_escape")
    if target.is_file():
        existing_size = target.stat().st_size
        existing_digest = _sha256(target)
        if ((clean.get("bytes") is None or existing_size == clean["bytes"]) and
                (not clean.get("sha256") or existing_digest == clean["sha256"].lower())):
            extracted = []
            if extract and clean["format"].lower() == "zip":
                extracted = _safe_extract_zip(target, target_dir / clean["id"], max_bytes)
            return {"schema_version": "mathmodel.external-dataset-receipt/v1", "id": clean["id"],
                    "path": str(target), "bytes": existing_size, "sha256": existing_digest,
                    "verified": bool(clean.get("sha256") and clean.get("bytes") is not None),
                    "source_url": clean["url"], "reused": True, "extracted_files": extracted}
        raise ExternalDatasetError("dataset_existing_mismatch")
    fd, temp_name = tempfile.mkstemp(prefix=f".{clean['id']}.", suffix=".part", dir=target_dir)
    os.close(fd)
    temp = Path(temp_name)
    size = 0
    try:
        request = Request(clean["url"], headers={"User-Agent": "Mathematical-Modeling/1.0"})
        with urlopen(request, timeout=timeout) as response, temp.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                _require(size <= max_bytes, "dataset_size_limit")
                handle.write(chunk)
        if clean.get("bytes") is not None:
            _require(size == clean["bytes"], "dataset_byte_size_mismatch")
        digest = _sha256(temp)
        if clean.get("sha256"):
            _require(digest == clean["sha256"].lower(), "dataset_sha256_mismatch")
        temp.replace(target)
        extracted: list[str] = []
        if extract and clean["format"].lower() == "zip":
            extracted = _safe_extract_zip(target, target_dir / clean["id"], max_bytes)
        return {"schema_version": "mathmodel.external-dataset-receipt/v1", "id": clean["id"],
                "path": str(target), "bytes": size, "sha256": digest,
                "verified": bool(clean.get("sha256") and clean.get("bytes") is not None),
                "source_url": clean["url"], "extracted_files": extracted}
    finally:
        if temp.exists():
            temp.unlink()


__all__ = ["SCHEMA", "ExternalDatasetError", "validate_catalog", "load_catalog",
           "dataset", "download_dataset"]
