"""Leakage-aware transformations for open-model benchmark cases.

The transformations change only the public case description.  They never
reuse a development data fingerprint as a structure-transform ground truth;
the caller must explicitly provide a new fingerprint obtained from an
independently generated or audited transformed fixture.
"""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Mapping

from .open_model_bench import BenchmarkCase, BenchmarkValidationError


def _clean_map(values: Mapping[str, str] | None, *, code: str) -> dict[str, str]:
    if values is None:
        return {}
    if not isinstance(values, Mapping) or len(values) > 32:
        raise BenchmarkValidationError(code)
    result = {}
    for old, new in values.items():
        if (type(old) is not str or type(new) is not str or not old.strip()
                or not new.strip() or len(old) > 160 or len(new) > 160):
            raise BenchmarkValidationError(code)
        result[old] = new
    return result


def transform_statement(
    statement: str,
    *,
    symbol_map: Mapping[str, str] | None = None,
    story_map: Mapping[str, str] | None = None,
    unit_map: Mapping[str, str] | None = None,
) -> str:
    """Apply bounded, word-aware replacements without executing formulas."""
    if type(statement) is not str or not statement.strip() or len(statement) > 64_000:
        raise BenchmarkValidationError("invalid_transform_statement")
    symbols = _clean_map(symbol_map, code="invalid_symbol_transform")
    stories = _clean_map(story_map, code="invalid_story_transform")
    units = _clean_map(unit_map, code="invalid_unit_transform")
    output = statement
    for old, new in sorted(symbols.items(), key=lambda item: (-len(item[0]), item[0])):
        output = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])",
                        lambda _match, replacement=new: replacement, output)
    for old, new in sorted(stories.items(), key=lambda item: (-len(item[0]), item[0])):
        output = output.replace(old, new)
    # Units are token-like but may contain punctuation (e.g. ``m/s``); replace
    # only when surrounded by non-word characters to avoid changing identifiers.
    for old, new in sorted(units.items(), key=lambda item: (-len(item[0]), item[0])):
        output = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])",
                        lambda _match, replacement=new: replacement, output)
    if not output.strip() or len(output) > 64_000:
        raise BenchmarkValidationError("invalid_transformed_statement")
    return output


def transform_case(
    case: BenchmarkCase,
    *,
    transform_id: str,
    symbol_map: Mapping[str, str] | None = None,
    story_map: Mapping[str, str] | None = None,
    unit_map: Mapping[str, str] | None = None,
    transformed_data_fingerprints: tuple[str, ...] = (),
    has_ground_truth: bool = False,
) -> BenchmarkCase:
    """Create an explicitly marked structure-transform case.

    The default has no ground truth and no data fingerprint.  Setting
    ``has_ground_truth`` requires at least one *new* fingerprint, preventing a
    transformed statement from silently sharing the original fixture.
    """
    if not isinstance(case, BenchmarkCase):
        raise TypeError("benchmark_case_required")
    if type(transform_id) is not str or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", transform_id):
        raise BenchmarkValidationError("invalid_transform_id")
    if type(has_ground_truth) is not bool:
        raise BenchmarkValidationError("invalid_transformed_ground_truth_flag")
    fingerprints = tuple(transformed_data_fingerprints)
    if set(fingerprints) & set(case.data_fingerprints):
        raise BenchmarkValidationError("transformed_data_reuses_original_fingerprint")
    transformed = replace(
        case,
        case_id=f"{case.case_id}__{transform_id}",
        split="structure_transform",
        statement=transform_statement(case.statement, symbol_map=symbol_map, story_map=story_map, unit_map=unit_map),
        data_fingerprints=fingerprints,
        tags=tuple(dict.fromkeys((*case.tags, f"transform:{transform_id}"))),
        has_ground_truth=has_ground_truth,
    )
    if transformed.has_ground_truth and not fingerprints:
        raise BenchmarkValidationError("transformed_ground_truth_requires_new_data")
    # Re-run the case contract's normal validation, including fingerprint shape.
    return BenchmarkCase.from_payload(transformed.public())


__all__ = ["transform_statement", "transform_case"]
