"""Controls for measuring false latent-state suggestions.

The benchmark deliberately treats delay/PCA compression as a mathematical
candidate, not a discovered physical variable.  A high-noise independent
control is used to estimate how often the heuristic would suggest compression
without a known latent mechanism.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import numpy as np

from .latent_state_discovery import discover_delay_latent_states


class LatentBenchmarkError(ValueError):
    pass


def assess_latent_false_positive_control(
    generator: Callable[[int, int], tuple[np.ndarray, np.ndarray]], *,
    replicates: int = 8, seed: int = 0, latent_threshold: float = 0.8,
) -> dict[str, Any]:
    """Run a bounded no-latent control and report heuristic flag frequency.

    The generator receives ``(seed, replicate)`` and returns strictly increasing
    times and an observation matrix. A flag means the delay candidate selected
    fewer states than observed while explaining at least ``latent_threshold``
    of its training variance. This is an operating characteristic, not a
    hypothesis test or proof that a real latent state exists.
    """
    if not callable(generator):
        raise LatentBenchmarkError("generator_must_be_callable")
    if type(replicates) is not int or not 2 <= replicates <= 256:
        raise LatentBenchmarkError("replicates_out_of_bounds")
    if type(seed) is not int:
        raise LatentBenchmarkError("seed_must_be_integer")
    if not math.isfinite(float(latent_threshold)) or not 0 < float(latent_threshold) < 1:
        raise LatentBenchmarkError("invalid_latent_threshold")
    rows = []
    for replicate in range(replicates):
        try:
            times, observations = generator(seed, replicate)
            matrix = np.asarray(observations, dtype=float)
            result = discover_delay_latent_states(times, matrix, latent_dim=None)
            observed_dim = int(matrix.shape[1])
            selected = int(result["latent_dim"])
            explained = float(sum(result["explained_variance"]))
            flag = selected < observed_dim and explained >= float(latent_threshold)
            rows.append({"replicate": replicate, "status": "ok", "observed_dim": observed_dim,
                         "selected_dim": selected, "explained_variance": explained,
                         "false_positive_flag": flag})
        except Exception as exc:
            rows.append({"replicate": replicate, "status": "not_assessed",
                         "error_code": type(exc).__name__})
    assessed = [row for row in rows if row["status"] == "ok"]
    flags = sum(bool(row["false_positive_flag"]) for row in assessed)
    return {
        "schema_version": "mathmodel.latent-state-benchmark/v1",
        "status": "assessed" if len(assessed) == replicates else "partial",
        "replicates": replicates, "assessed_replicates": len(assessed),
        "false_positive_count": flags,
        "false_positive_rate": flags / len(assessed) if assessed else None,
        "rows": rows,
        "policy": "no_latent_control_measures_heuristic_compression_flags_not_physical_latent_truth",
    }


__all__ = ["LatentBenchmarkError", "assess_latent_false_positive_control"]
