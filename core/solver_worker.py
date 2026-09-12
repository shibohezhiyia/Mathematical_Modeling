"""Private entry point: JSON contracts in, bounded numeric JSON out.

Only trusted built-in executors are reachable. Do not add an exec/eval/module
parameter here; OS resource limits do not isolate filesystem/network access.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

# -I excludes cwd and PYTHONPATH. Add only this trusted installation's root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.solver_runtime import (
    EvaluationBudgetExceeded, MAX_INPUT_BYTES, PROTOCOL, SolverLimits,
    SolverRuntimeError, decode_message, encode_message,
)
from core.solver_process_limits import ResourceIsolationUnavailable, install_linux_limits
from core.worker_permissions import WorkerPermissionError, WorkerPermissionPolicy, make_audit_hook


def _install_worker_permissions(directory: str) -> None:
    """Install a defense-in-depth audit hook before numerical imports."""
    trusted = {str(Path(directory).resolve()), str(Path(__file__).resolve().parents[1]),
               str(Path(sys.prefix).resolve()), str(Path(sys.base_prefix).resolve()),
               str(Path(sys.executable).resolve().parent)}
    policy = WorkerPermissionPolicy(tuple(sorted(trusted)), allow_network=False,
                                    allow_subprocess=False).validate()
    sys.addaudithook(make_audit_hook(policy))


def main() -> None:
    limits = SolverLimits()
    try:
        # On Windows the parent attaches the job before releasing stdin.
        request = decode_message(sys.stdin.buffer.read(MAX_INPUT_BYTES + 1), MAX_INPUT_BYTES)
        if set(request) != {"protocol", "executor_key", "contract", "limits"} or request["protocol"] != PROTOCOL:
            raise SolverRuntimeError("invalid_contract")
        limits = SolverLimits(**request["limits"])
        if os.name != "nt":
            install_linux_limits(limits.memory_mb * 1024 * 1024)
        # Load trusted numerical runtimes before the audit hook.  Some BLAS/
        # platform-discovery code opens their own installation files or asks
        # the OS for version information during import; those are not user
        # operations and must not be mistaken for generated-code access.
        import numpy  # noqa: F401
        try:
            import scipy  # noqa: F401
            # Import lazy SciPy namespaces now; some of them trigger NumPy's
            # platform probe, which legitimately uses an OS subprocess.
            import scipy.integrate  # noqa: F401
            import scipy.linalg  # noqa: F401
            import scipy.optimize  # noqa: F401
            import scipy.stats  # noqa: F401
        except ImportError:
            pass
        try:
            _install_worker_permissions(os.environ.get("TMPDIR") or os.getcwd())
        except WorkerPermissionError as exc:
            raise SolverRuntimeError("permission_isolation_unavailable") from exc

        # Numerical imports happen only after mandatory resource setup.
        if request["executor_key"] in ("scalar_graph/v1", "scalar_graph_confirm/v1"):
            from core.graph_evaluator import evaluate_graph_request
            from core.graph_confirmation import evaluate_frozen_request
            from core.model_hypotheses import HypothesisValidationError
            try:
                evaluate = evaluate_frozen_request if request["executor_key"] == "scalar_graph_confirm/v1" else evaluate_graph_request
                result = evaluate(request["contract"], max_evaluations=limits.max_evaluations)
            except HypothesisValidationError as exc:
                raise SolverRuntimeError("invalid_contract") from exc
        elif request["executor_key"] in (
            "linear_ode/v1", "threshold_event/v1", "quadratic_program/v1",
            "distance/v1", "interval_union/v1", "region_membership/v1", "segment_intersection/v1",
            "line_of_sight/v1", "normal_log_likelihood/v1", "bootstrap_mean/v1", "permutation_test/v1",
        ):
            # These primitive routes contain no user callback or generated
            # code.  Keep the allow-list explicit so adding a registry entry
            # cannot accidentally expose every executor to the process worker.
            from core.universal_math_solvers import UniversalSolverRegistry

            contract = request["contract"]
            if type(contract) is not dict:
                raise SolverRuntimeError("invalid_contract")
            result = UniversalSolverRegistry().execute(request["executor_key"], contract)
        else:
            from core.mechanistic_modeling import MechanisticModelingEngine
            routes = {
                "adaptive_ode/v1": ("ode_system", MechanisticModelingEngine._solve_ode_system),
                "bounded_nlp/v1": ("optimization_problem", MechanisticModelingEngine._solve_optimization_problem),
            }
            route = routes.get(request["executor_key"])
            contract = request["contract"]
            if route is None or type(contract) is not dict or contract.get("kind") != route[0]:
                raise SolverRuntimeError("invalid_contract")
            verified = MechanisticModelingEngine._verify_structured_relation(contract)
            if verified.get("parse_status") != "machine_verified":
                raise SolverRuntimeError("invalid_contract")
            result = route[1](verified, max_evaluations=limits.max_evaluations)
        response = encode_message({"protocol": PROTOCOL, "status": "ok", "result": result}, limits.output_bytes)
    except Exception as exc:
        if isinstance(exc, MemoryError):
            code = "memory_limit"
        elif isinstance(exc, EvaluationBudgetExceeded):
            code = "evaluation_limit"
        elif isinstance(exc, ResourceIsolationUnavailable):
            code = "isolation_unavailable"
        elif isinstance(exc, SolverRuntimeError):
            code = exc.code
        elif isinstance(exc, (ZeroDivisionError, OverflowError, FloatingPointError)):
            code = "numeric_domain"
        else:
            code = "solver_failure"
        # No tracebacks, paths, environment or input excerpts in the response.
        response = encode_message({"protocol": PROTOCOL, "status": "failed", "code": code}, limits.output_bytes)
    sys.stdout.buffer.write(response)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
