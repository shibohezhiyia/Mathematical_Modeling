"""Adapter for the official SRSD-Feynman split repositories."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .llm_srbench_adapter import run_llm_srbench_pilot


class SRSDAdapterError(ValueError):
    pass


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _arity(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline().strip().split()
    if len(first) < 2:
        raise SRSDAdapterError("dataset_row_invalid")
    return len(first) - 1


def select_srsd_instances(dataset_dir: str | Path, *, count: int, seed: int,
                          exclude: Sequence[str] = (), maximum_arity: int = 4) -> tuple[str, ...]:
    root = Path(dataset_dir).resolve()
    if type(count) is not int or count < 1 or type(seed) is not int or maximum_arity < 1:
        raise SRSDAdapterError("selection_configuration_invalid")
    candidates = []
    for path in sorted((root / "train").glob("*.txt")):
        instance_id = path.stem
        if instance_id not in set(exclude) and _arity(path) <= maximum_arity:
            rank = sha256(f"{seed}:{instance_id}:{_arity(path)}".encode()).hexdigest()
            candidates.append((rank, instance_id))
    if len(candidates) < count:
        raise SRSDAdapterError("insufficient_supported_instances")
    return tuple(item[1] for item in sorted(candidates)[:count])


def load_srsd_cases(dataset_dir: str | Path, *, instance_ids: Sequence[str],
                    source_revision: str, train_limit: int = 512,
                    test_limit: int = 256, seed: int = 20261008) -> tuple[dict[str, Any], ...]:
    root = Path(dataset_dir).resolve()
    if (not instance_ids or len(set(instance_ids)) != len(instance_ids)
            or len(source_revision) not in {40, 64}):
        raise SRSDAdapterError("case_request_invalid")
    if not 32 <= train_limit <= 4096 or not 16 <= test_limit <= 2048:
        raise SRSDAdapterError("sample_limit_invalid")
    info_path, readme_path = root / "supp_info.json", root / "README.md"
    if not info_path.is_file() or not readme_path.is_file():
        raise SRSDAdapterError("dataset_metadata_missing")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    result = []
    for ordinal, instance_id in enumerate(instance_ids):
        train_path = root / "train" / f"{instance_id}.txt"
        test_path = root / "test" / f"{instance_id}.txt"
        equation_path = root / "true_eq" / f"{instance_id}.pkl"
        if not train_path.is_file() or not test_path.is_file() or instance_id not in info:
            raise SRSDAdapterError("requested_instance_missing")
        train = np.loadtxt(train_path, dtype=float)
        test = np.loadtxt(test_path, dtype=float)
        if (train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]
                or not np.isfinite(train).all() or not np.isfinite(test).all()
                or not 1 <= train.shape[1] - 1 <= 4):
            raise SRSDAdapterError("dataset_values_invalid")
        rng = np.random.default_rng(seed + ordinal)
        train_indices = np.sort(rng.choice(len(train), min(train_limit, len(train)), replace=False))
        test_indices = np.sort(rng.choice(len(test), min(test_limit, len(test)), replace=False))
        variables = [f"x{index}" for index in range(train.shape[1] - 1)]
        rows = [{**{name: float(train[row_index, column_index])
                    for column_index, name in enumerate(variables)},
                 "response": float(train[row_index, -1])} for row_index in train_indices]
        files = [train_path, test_path, equation_path, info_path, readme_path]
        manifest = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                     "sha256": _digest(path)} for path in files]
        result.append({
            "instance_id": instance_id, "subset": "srsd-feynman_easy",
            "public_input": {
                "problem": "Infer the numeric relationship from the supplied SRSD observations.",
                "attachments": [{"name": "srsd_observations", "format": "records", "rows": rows}],
                "query_inputs": test[test_indices, :-1].tolist(),
            },
            "hidden_reference": {
                "test_output": test[test_indices, -1].tolist(),
                "ground_truth_expression": str(info[instance_id]["sympy_eq_str"]),
            },
            "source": {"repository": "yoshitomo-matsubara/srsd-feynman_easy",
                       "relationship": "official_srsd_huggingface_dataset",
                       "revision": source_revision, "license": "CC-BY-4.0", "files": manifest},
        })
    return tuple(result)


def run_srsd_pilot(cases: Sequence[Mapping[str, Any]], *, solver=None,
                   status: str = "development_only_srsd_official") -> dict[str, Any]:
    report = run_llm_srbench_pilot(
        cases, solver=solver, status=status,
        selection_policy="hash_selected_before_truth_load;development_only",
        source_policy="official_srsd_snapshot;cc_by_4_0",
    )
    report["schema_version"] = "mathmodel.srsd-pilot/v1"
    return report


__all__ = ["SRSDAdapterError", "select_srsd_instances", "load_srsd_cases", "run_srsd_pilot"]
