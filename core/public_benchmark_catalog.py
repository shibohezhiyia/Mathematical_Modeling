"""公开数学建模题源目录（不复制题面，也不包含参考解）。

目录只解决两个可复核问题：题目来自哪里，以及封存案例如何和官方页面对应。
题面、附件和参考评分标准仍由 ``blind_benchmark`` 的封存目录管理；本模块
不会下载、解析或执行网页内容，也不会把网页摘要当作答案证据。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

CATALOG_SCHEMA = "mathmodel.public-benchmark-catalog/v1"
OFFICIAL_HOSTS = frozenset({
    "www.contest.comap.com", "contest.comap.com", "www.comap.com",
    "www.mcm.edu.cn", "mcm.edu.cn", "www.cmathc.org.cn", "dxs.moe.gov.cn",
})
TRACKS = frozenset({"MCM", "ICM", "CUMCM"})
AVAILABILITY = frozenset({"source_listed", "sealed_locally", "needs_manual_review"})


class PublicCatalogError(ValueError):
    """公开题源目录契约错误。``code`` 稳定且不包含网页正文。"""
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise PublicCatalogError(code)


def _text(value: Any, *, maximum: int = 240) -> str:
    if type(value) is not str or not value.strip() or len(value) > maximum:
        _fail("invalid_text")
    return value.strip()


def _url(value: Any) -> str:
    value = _text(value, maximum=2_000)
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        _fail("url_must_be_https")
    host = parsed.hostname.lower() if parsed.hostname else ""
    if host not in OFFICIAL_HOSTS:
        _fail("unapproved_source_host")
    return value


def _tags(value: Any) -> tuple[str, ...]:
    if type(value) is not list or not value or len(value) > 24:
        _fail("invalid_tags")
    result = []
    for item in value:
        item = _text(item, maximum=64)
        if any(char.isspace() for char in item) or item in result:
            _fail("invalid_tags")
        result.append(item)
    return tuple(result)


@dataclass(frozen=True)
class PublicBenchmarkSource:
    """官方来源与封存案例的公开映射，不含题面、附件或答案内容。"""
    source_id: str
    case_id: str
    provider: str
    year: int
    track: str
    problem: str
    source_url: str
    statement_url: str
    tags: tuple[str, ...]
    availability: str = "source_listed"
    retrieved_on: str = ""
    attachment_urls: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PublicBenchmarkSource":
        if type(payload) is not dict:
            _fail("source_must_be_object")
        allowed = {"id", "case_id", "provider", "year", "track", "problem", "source_url",
                   "statement_url", "tags", "availability", "retrieved_on", "attachment_urls"}
        required = {"id", "case_id", "provider", "year", "track", "problem", "source_url",
                    "statement_url", "tags"}
        if set(payload) - allowed or not required <= set(payload):
            _fail("invalid_source_fields")
        source_id, case_id = _text(payload["id"], maximum=120), _text(payload["case_id"], maximum=128)
        if any(char.isspace() for char in source_id + case_id):
            _fail("invalid_source_id")
        provider = _text(payload["provider"], maximum=80)
        year = payload["year"]
        if type(year) is not int or not 1990 <= year <= date.today().year + 1:
            _fail("invalid_year")
        track = payload["track"]
        if track not in TRACKS:
            _fail("invalid_track")
        problem = _text(payload["problem"], maximum=80)
        source_url, statement_url = _url(payload["source_url"]), _url(payload["statement_url"])
        tags = _tags(payload["tags"])
        availability = payload.get("availability", "source_listed")
        if availability not in AVAILABILITY:
            _fail("invalid_availability")
        retrieved_on = payload.get("retrieved_on", "")
        if retrieved_on:
            try:
                date.fromisoformat(_text(retrieved_on, maximum=10))
            except ValueError as exc:
                raise PublicCatalogError("invalid_retrieved_on") from exc
        attachments = payload.get("attachment_urls", [])
        if type(attachments) is not list or len(attachments) > 16:
            _fail("invalid_attachment_urls")
        attachment_urls = tuple(_url(item) for item in attachments)
        forbidden = ("answer", "solution", "reference_answer", "ground_truth", "raw_data")
        if any(token in json.dumps(payload, ensure_ascii=False).lower() for token in forbidden):
            _fail("solution_material_forbidden")
        return cls(source_id, case_id, provider, year, track, problem, source_url, statement_url,
                   tags, availability, retrieved_on, attachment_urls)

    def public(self) -> dict[str, Any]:
        return {"id": self.source_id, "case_id": self.case_id, "provider": self.provider,
                "year": self.year, "track": self.track, "problem": self.problem,
                "source_url": self.source_url, "statement_url": self.statement_url,
                "tags": list(self.tags), "availability": self.availability,
                "retrieved_on": self.retrieved_on, "attachment_urls": list(self.attachment_urls)}


@dataclass(frozen=True)
class PublicBenchmarkCatalog:
    _json: str

    @classmethod
    def create(cls, sources: Iterable[PublicBenchmarkSource | Mapping[str, Any]], *, name: str,
               revision: int = 1) -> "PublicBenchmarkCatalog":
        if type(name) is not str or not name.strip() or len(name) > 120:
            _fail("invalid_catalog_name")
        if type(revision) is not int or revision < 1:
            _fail("invalid_revision")
        try:
            parsed = [item if isinstance(item, PublicBenchmarkSource) else PublicBenchmarkSource.from_payload(item)
                      for item in sources]
        except TypeError as exc:
            raise PublicCatalogError("invalid_source_count") from exc
        if not parsed or len(parsed) > 10_000:
            _fail("invalid_source_count")
        ids, cases = [item.source_id for item in parsed], [item.case_id for item in parsed]
        if len(ids) != len(set(ids)):
            _fail("duplicate_source_id")
        if len(cases) != len(set(cases)):
            _fail("duplicate_case_id")
        payload = {"schema_version": CATALOG_SCHEMA, "name": name.strip(), "revision": revision,
                   "sources": [item.public() for item in sorted(parsed, key=lambda value: value.source_id)],
                   "policy": {"contains_problem_text": False, "contains_answers": False,
                              "source_pages_are_not_evidence_of_correctness": True}}
        return cls(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PublicBenchmarkCatalog":
        if type(payload) is not dict or set(payload) != {"schema_version", "name", "revision", "sources", "policy"}:
            _fail("invalid_catalog_fields")
        if payload["schema_version"] != CATALOG_SCHEMA:
            _fail("schema_mismatch")
        catalog = cls.create(payload["sources"], name=payload["name"], revision=payload["revision"])
        if payload["policy"] != json.loads(catalog._json)["policy"]:
            _fail("invalid_catalog_policy")
        return catalog

    @property
    def digest(self) -> str:
        return hashlib.sha256(self._json.encode("utf-8")).hexdigest()

    def public(self) -> dict[str, Any]:
        return json.loads(self._json)

    def sources(self, *, track: str | None = None, year: int | None = None) -> tuple[PublicBenchmarkSource, ...]:
        if track is not None and track not in TRACKS:
            _fail("invalid_track")
        if year is not None and (type(year) is not int or year < 1990):
            _fail("invalid_year")
        return tuple(PublicBenchmarkSource.from_payload(item) for item in self.public()["sources"]
                     if (track is None or item["track"] == track) and (year is None or item["year"] == year))


def load_public_catalog(path: str | Path) -> PublicBenchmarkCatalog:
    """读取公开目录；不会访问网络。"""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicCatalogError("catalog_unreadable") from exc
    return PublicBenchmarkCatalog.from_payload(payload)


__all__ = ["CATALOG_SCHEMA", "PublicCatalogError", "PublicBenchmarkSource",
           "PublicBenchmarkCatalog", "load_public_catalog"]
