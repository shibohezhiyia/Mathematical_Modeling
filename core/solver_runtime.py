"""Bounded process execution for explicitly allow-listed, trusted solvers.

No generated code, pickle, command, module path or callable crosses this API.
This is resource isolation, not a filesystem/network permission boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

from .solver_process_limits import ResourceIsolationUnavailable, WindowsJob, resource_backend
from .shared_resource_quota import SharedResourceQuota


SUPPORTED_EXECUTORS = frozenset({
    "adaptive_ode/v1", "bounded_nlp/v1", "scalar_graph/v1", "scalar_graph_confirm/v1",
    "linear_ode/v1", "threshold_event/v1", "quadratic_program/v1",
    "distance/v1", "interval_union/v1", "region_membership/v1", "segment_intersection/v1",
    "line_of_sight/v1", "normal_log_likelihood/v1", "bootstrap_mean/v1", "permutation_test/v1",
})
PROTOCOL = "mathmodel.solver-process/v1"
MAX_INPUT_BYTES = 1_048_576
MAX_OUTPUT_BYTES = 16_777_216
_SLOTS = threading.BoundedSemaphore(2)  # Per server process, not a cluster-wide quota.
_MEMORY_CAP_MB = 2048  # Aggregate declared worker budget for this server process.
_MEMORY_CONDITION = threading.Condition()
_MEMORY_RESERVED_MB = 0
_CONTAINER_RECOVERY_REQUIRED = threading.Event()
_FAILURE_DETAILS = {
    "timeout": ("计算超过强制时限", "检查刚性、尺度和求解规模；不能因此判定模型错误。"),
    "queue_timeout": ("等待计算资源超时", "稍后重试，或减少同时运行的研究任务。"),
    "memory_limit": ("计算达到内存上限", "检查数组规模，考虑稀疏或分块表示；不能擅自删除约束。"),
    "evaluation_limit": ("数值评估次数耗尽", "检查收敛与尺度，选择适合结构的算法；保留验证门槛。"),
    "numeric_domain": ("出现数值定义域或溢出错误", "检查除零、负数开方、参数范围与中间状态。"),
    "isolation_unavailable": ("无法建立强制资源限制", "检查操作系统支持和运行权限；本次不回退到主进程。"),
    "permission_isolation_unavailable": ("无法建立文件、网络或子进程权限边界", "检查 worker 权限策略和可信目录；本次不回退到主进程。"),
    "container_cleanup_failed": ("无法确认隔离容器已回收", "检查容器服务及该任务的 container_name；结果不发布，不要清理其他用户的容器。"),
    "container_recovery_required": ("隔离容器回收异常，已暂停新的严格任务", "核实并回收日志中该任务的容器后再重启服务；不要直接重试堆积容器。"),
    "cancelled": ("计算已取消", "本次不产生数值结论，可在确认输入后重新运行。"),
    "worker_exit": ("计算进程异常退出", "检查依赖与资源环境；退出原因未确定，不能当成数学反例。"),
    "output_limit": ("计算输出超过上限", "减少输出采样或摘要体积，不能截断后冒充完整结果。"),
    "disk_limit": ("计算临时目录超过磁盘配额", "减少中间文件或改用流式/分块算法；本次不产生数值结论。"),
    "invalid_contract": ("执行契约未通过复核", "补齐变量、单位和合法表达式；外部模型不能自授执行权限。"),
    "upstream_failed": ("上游无可用结果，已阻止执行", "先解决上游计算失败；不得使用占位值继续求解。"),
}


def failure_details(code: str) -> dict:
    label, action = _FAILURE_DETAILS.get(code, ("受控计算未完成", "检查契约或运行环境；本次不产生数值结论。"))
    return {"failure_label": label, "next_action": action, "mathematical_verdict": "not_assessed"}


class SolverRuntimeError(RuntimeError):
    def __init__(self, code: str, *, metadata: dict | None = None) -> None:
        self.code = code
        self.metadata = metadata or {}
        # Never propagate child stderr, a traceback, local paths or contract text.
        super().__init__(f"restricted numerical execution failed: {code}")


class EvaluationBudgetExceeded(RuntimeError):
    pass


class EvaluationCounter:
    def __init__(self, maximum: int) -> None:
        if type(maximum) is not int or not 1 <= maximum <= 1_000_000:
            raise ValueError("invalid evaluation budget")
        self.maximum = maximum
        self.used = 0

    def consume(self) -> None:
        if self.used >= self.maximum:
            raise EvaluationBudgetExceeded("numerical evaluation budget exhausted")
        self.used += 1


def _reserve_memory(memory_mb: int, timeout: float, cancel: threading.Event | None) -> bool:
    global _MEMORY_RESERVED_MB
    deadline = time.monotonic() + max(0.0, timeout)
    with _MEMORY_CONDITION:
        while _MEMORY_RESERVED_MB + memory_mb > _MEMORY_CAP_MB:
            if cancel is not None and cancel.is_set():
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            _MEMORY_CONDITION.wait(timeout=min(remaining, 0.05))
        _MEMORY_RESERVED_MB += memory_mb
        return True


def _release_memory(memory_mb: int) -> None:
    global _MEMORY_RESERVED_MB
    with _MEMORY_CONDITION:
        _MEMORY_RESERVED_MB = max(0, _MEMORY_RESERVED_MB - memory_mb)
        _MEMORY_CONDITION.notify_all()


@dataclass(frozen=True)
class SolverLimits:
    wall_seconds: float = 30.0
    memory_mb: int = 1024
    max_evaluations: int = 250_000
    output_bytes: int = MAX_OUTPUT_BYTES
    disk_bytes: int = 64 * 1024 * 1024
    isolation_mode: str = "python_audit"

    def __post_init__(self) -> None:
        if (type(self.wall_seconds) not in (int, float)
                or not math.isfinite(self.wall_seconds) or not 0.05 <= self.wall_seconds <= 120):
            raise ValueError("wall_seconds must be finite and between 0.05 and 120")
        for value, lower, upper in (
            (self.memory_mb, 64, 2048), (self.max_evaluations, 1, 1_000_000),
            (self.output_bytes, 1024, MAX_OUTPUT_BYTES),
            (self.disk_bytes, 1 * 1024 * 1024, 4 * 1024 * 1024 * 1024),
        ):
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError("invalid resource limit")
        if self.isolation_mode not in {"python_audit", "strict_os"}:
            raise ValueError("invalid isolation mode")


def _check_plain_json(value: Any) -> None:
    """Bound depth/size BEFORE encoding; no custom serialization or numeric hooks."""
    stack = [(value, 0)]
    count = 0
    while stack:
        item, depth = stack.pop()
        count += 1
        if count > 600_000 or depth > 24:
            raise SolverRuntimeError("invalid_payload")
        if type(item) is dict:
            if len(item) > 20_000 or any(type(key) is not str for key in item):
                raise SolverRuntimeError("invalid_payload")
            stack.extend((key, depth + 1) for key in item)
            stack.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            if len(item) > 20_000:
                raise SolverRuntimeError("invalid_payload")
            stack.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            if len(item) > MAX_INPUT_BYTES:
                raise SolverRuntimeError("invalid_payload")
        elif type(item) is int:
            if item.bit_length() > 1024:
                raise SolverRuntimeError("invalid_payload")
        elif type(item) is float:
            if not math.isfinite(item):
                raise SolverRuntimeError("nonfinite_result")
        elif item is not None and type(item) is not bool:
            raise SolverRuntimeError("invalid_payload")


def encode_message(payload: dict, maximum: int, *, limit_code: str = "output_limit") -> bytes:
    _check_plain_json(payload)
    # iterencode bounds the retained serialization; unlike dumps it does not
    # first allocate the entire response when a solver returns too much data.
    output = bytearray()
    for chunk in json.JSONEncoder(ensure_ascii=True, allow_nan=False, separators=(",", ":")).iterencode(payload):
        encoded = chunk.encode("ascii")
        if len(output) + len(encoded) > maximum:
            raise SolverRuntimeError(limit_code)
        output.extend(encoded)
    return bytes(output)


def decode_message(raw: bytes, maximum: int) -> dict:
    if len(raw) > maximum:
        raise SolverRuntimeError("invalid_payload")
    # Prevent the decoder's own deep recursion before JSON validation.
    depth, quoted, escaped = 0, False, False
    for char in raw:
        if quoted:
            if escaped:
                escaped = False
            elif char == 92:
                escaped = True
            elif char == 34:
                quoted = False
        elif char == 34:
            quoted = True
        elif char in (91, 123):
            depth += 1
            if depth > 24:
                raise SolverRuntimeError("invalid_payload")
        elif char in (93, 125):
            depth -= 1

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        payload = json.loads(raw, object_pairs_hook=pairs,
                             parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite")))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise SolverRuntimeError("invalid_payload") from exc
    if type(payload) is not dict:
        raise SolverRuntimeError("invalid_payload")
    _check_plain_json(payload)
    return payload


def _worker_environment(directory: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key.upper() in {"SYSTEMROOT", "WINDIR"}}
    env.update({
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.defpath,
        "TMP": directory, "TEMP": directory, "TMPDIR": directory,
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    })
    return env


def _worker_command() -> list[str]:
    return [sys.executable, "-I", "-B", "-u", str(Path(__file__).with_name("solver_worker.py"))]


def _directory_size_exceeds(directory: str, limit_bytes: int) -> bool:
    """Bound temporary worker storage without following links outside it."""
    total = 0
    root = Path(directory).resolve()
    try:
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                total += path.stat().st_size
            except OSError:
                continue
            if total > limit_bytes:
                return True
    except OSError:
        return False
    return False


class _BoundedPipe:
    def __init__(self, pipe, limit: int) -> None:
        self.pipe, self.limit = pipe, limit
        self.data = bytearray()
        self.exceeded = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        try:
            while True:
                chunk = self.pipe.read(8192)
                if not chunk:
                    return
                remaining = self.limit - len(self.data)
                self.data.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    self.exceeded.set()
                    return
        except (OSError, ValueError):
            return


class SolverProcessRunner:
    """Disposable worker per contract, capped concurrency, no unsafe fallback."""

    def execute(self, executor_key: str, contract: dict, *, limits: SolverLimits | None = None,
                cancel: threading.Event | None = None,
                shared_quota: SharedResourceQuota | None = None) -> dict:
        if executor_key not in SUPPORTED_EXECUTORS:
            raise SolverRuntimeError("unsupported_executor")
        limits = limits or SolverLimits()
        if limits.isolation_mode == "strict_os":
            if _CONTAINER_RECOVERY_REQUIRED.is_set():
                raise SolverRuntimeError("container_recovery_required")
            from .os_sandbox import require_strict_os_sandbox
            try:
                require_strict_os_sandbox()
            except Exception as exc:
                raise SolverRuntimeError("permission_isolation_unavailable") from exc
        started = time.monotonic()
        request = encode_message({
            "protocol": PROTOCOL, "executor_key": executor_key,
            "contract": contract, "limits": asdict(limits),
        }, MAX_INPUT_BYTES, limit_code="input_limit")
        metadata = {
            "protocol": PROTOCOL, "process_isolated": False,
            "permission_isolated": False, "arbitrary_code_allowed": False,
            "limits": asdict(limits), "blas_threads_requested": 1,
            "shared_quota": "disabled" if shared_quota is None else "enabled",
        }
        # Reject explicitly typed malformed mechanistic contracts before
        # starting a worker.  The worker still re-verifies every contract;
        # this preflight only prevents invalid requests from paying the costly
        # NumPy/SciPy worker import time and then being misreported as timeout.
        # Contracts without a kind remain worker-owned so probe/cancellation
        # paths keep their existing semantics.
        if executor_key in {"adaptive_ode/v1", "bounded_nlp/v1"} and isinstance(contract, dict) and "kind" in contract:
            expected_kind = "ode_system" if executor_key == "adaptive_ode/v1" else "optimization_problem"
            if contract.get("kind") != expected_kind:
                raise SolverRuntimeError("invalid_contract", metadata=metadata)
            try:
                from .mechanistic_modeling import MechanisticModelingEngine
                verified = MechanisticModelingEngine._verify_structured_relation(contract)
            except Exception as exc:
                raise SolverRuntimeError("invalid_contract", metadata=metadata) from exc
            if verified.get("parse_status") != "machine_verified":
                raise SolverRuntimeError("invalid_contract", metadata=metadata)
        # Both mechanistic routes perform at least a confirmation/multistart
        # evaluation after compilation.  A budget below two cannot execute a
        # valid contract, so report the deterministic budget failure before a
        # worker import can consume the entire wall-clock allowance.
        if executor_key in {"adaptive_ode/v1", "bounded_nlp/v1"} and limits.max_evaluations < 2:
            raise SolverRuntimeError("evaluation_limit", metadata=metadata)
        try:
            metadata["memory_backend"] = (
                "container_cgroup" if limits.isolation_mode == "strict_os" else resource_backend()
            )
        except ResourceIsolationUnavailable as exc:
            raise SolverRuntimeError("isolation_unavailable", metadata=metadata) from exc
        acquired = False
        memory_acquired = False
        shared_lease = None
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    raise SolverRuntimeError("cancelled", metadata=metadata)
                remaining = limits.wall_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise SolverRuntimeError("queue_timeout", metadata=metadata)
                if not acquired:
                    acquired = _SLOTS.acquire(timeout=min(remaining, 0.05))
                    if not acquired:
                        continue
                remaining = limits.wall_seconds - (time.monotonic() - started)
                if not _reserve_memory(limits.memory_mb, remaining, cancel):
                    if cancel is not None and cancel.is_set():
                        raise SolverRuntimeError("cancelled", metadata=metadata)
                    raise SolverRuntimeError("queue_timeout", metadata=metadata)
                memory_acquired = True
                if shared_quota is not None:
                    shared_lease = shared_quota.acquire(
                        f"solver-{os.getpid()}-{secrets.token_hex(8)}",
                        memory_mb=limits.memory_mb,
                        slots=1,
                        ttl_seconds=max(1.0, limits.wall_seconds + 5.0),
                    )
                    if shared_lease is None:
                        _release_memory(limits.memory_mb)
                        memory_acquired = False
                        _SLOTS.release()
                        acquired = False
                        if cancel is not None:
                            cancel.wait(min(remaining, 0.05))
                        else:
                            time.sleep(min(remaining, 0.05))
                        continue
                    metadata["shared_quota_lease"] = "acquired"
                else:
                    metadata["shared_quota_lease"] = "not_required"
                break
            with tempfile.TemporaryDirectory(prefix="mathmodel-solver-") as directory:
                return self._run(request, directory, limits, cancel, started, metadata)
        finally:
            metadata["elapsed_seconds"] = round(time.monotonic() - started, 6)
            if acquired:
                _SLOTS.release()
            if memory_acquired:
                _release_memory(limits.memory_mb)
            if shared_lease is not None and shared_lease.get("token") is not None and shared_quota is not None:
                try:
                    shared_quota.release(shared_lease["token"])
                except Exception:
                    metadata["shared_quota_release_warning"] = True

    @staticmethod
    def _run(request, directory, limits, cancel, started, metadata) -> dict:
        process, job, writer = None, None, None
        container = None
        readers = []
        from .os_sandbox import ContainerWorkerSession, OSSandboxError, OSSandboxCancelled
        try:
            if os.name == "nt" and limits.isolation_mode != "strict_os":
                job = WindowsJob(limits.memory_mb * 1024 * 1024)
            command = _worker_command()
            environment = _worker_environment(directory)
            if limits.isolation_mode == "strict_os":
                if _CONTAINER_RECOVERY_REQUIRED.is_set():
                    raise SolverRuntimeError("container_recovery_required", metadata=metadata)
                container = ContainerWorkerSession(memory_mb=limits.memory_mb, disk_bytes=limits.disk_bytes)
                metadata["container_image"] = container.image
                metadata["container_name"] = container.name
                remaining = limits.wall_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise SolverRuntimeError("timeout", metadata=metadata)
                command = container.prepare(remaining, cancel=cancel)
                environment = container.environment
            if cancel is not None and cancel.is_set():
                raise SolverRuntimeError("cancelled", metadata=metadata)
            if time.monotonic() - started >= limits.wall_seconds:
                raise SolverRuntimeError("timeout", metadata=metadata)
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=directory, env=environment, shell=False, bufsize=0,
                close_fds=True, start_new_session=(os.name != "nt"),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            metadata["process_isolated"] = True
            metadata["os_supervised"] = limits.isolation_mode == "strict_os"
            if job is not None:
                job.attach(process)
            stdout = _BoundedPipe(process.stdout, limits.output_bytes)
            stderr = _BoundedPipe(process.stderr, 65_536)
            readers = [stdout, stderr]

            def send_input():
                try:
                    # This is the release gate: no contract arrives before job attachment.
                    view = memoryview(request)
                    while view:
                        sent = process.stdin.write(view)
                        if not sent:
                            break
                        view = view[sent:]
                except (BrokenPipeError, OSError, ValueError):
                    pass
                finally:
                    process.stdin.close()

            writer = threading.Thread(target=send_input, daemon=True)
            writer.start()
            while True:
                if cancel is not None and cancel.is_set():
                    raise SolverRuntimeError("cancelled", metadata=metadata)
                if stdout.exceeded.is_set() or stderr.exceeded.is_set():
                    raise SolverRuntimeError("output_limit", metadata=metadata)
                if _directory_size_exceeds(directory, limits.disk_bytes):
                    metadata["disk_limit_bytes"] = limits.disk_bytes
                    raise SolverRuntimeError("disk_limit", metadata=metadata)
                if time.monotonic() - started >= limits.wall_seconds:
                    raise SolverRuntimeError("timeout", metadata=metadata)
                if process.poll() is not None and all(not reader.thread.is_alive() for reader in readers):
                    break
                # Event.wait permits prompt cancellation without a busy loop.
                if cancel is not None:
                    cancel.wait(0.02)
                else:
                    time.sleep(0.02)
            metadata["elapsed_seconds"] = round(time.monotonic() - started, 6)
            metadata["exit_code"] = process.returncode
            if process.returncode != 0:
                raise SolverRuntimeError("worker_exit", metadata=metadata)
            response = decode_message(bytes(stdout.data), limits.output_bytes)
            if cancel is not None and cancel.is_set():
                raise SolverRuntimeError("cancelled", metadata=metadata)
            if time.monotonic() - started >= limits.wall_seconds:
                raise SolverRuntimeError("timeout", metadata=metadata)
            if response.get("protocol") != PROTOCOL:
                raise SolverRuntimeError("invalid_response", metadata=metadata)
            if response.get("status") == "failed":
                code = response.get("code")
                if code not in {
                    "invalid_contract", "evaluation_limit", "memory_limit", "numeric_domain",
                    "nonfinite_result", "solver_failure", "output_limit", "isolation_unavailable",
                    "permission_isolation_unavailable",
                }:
                    code = "worker_failure"
                raise SolverRuntimeError(code, metadata=metadata)
            if response.get("status") != "ok" or type(response.get("result")) is not dict:
                raise SolverRuntimeError("invalid_response", metadata=metadata)
            result = response["result"]
            metadata["permission_isolated"] = True
            result["execution_supervision"] = metadata
            return result
        except ResourceIsolationUnavailable as exc:
            raise SolverRuntimeError("isolation_unavailable", metadata=metadata) from exc
        except OSSandboxCancelled as exc:
            raise SolverRuntimeError("cancelled", metadata=metadata) from exc
        except OSSandboxError as exc:
            raise SolverRuntimeError("permission_isolation_unavailable", metadata=metadata) from exc
        except subprocess.TimeoutExpired as exc:
            raise SolverRuntimeError("timeout", metadata=metadata) from exc
        except OSError as exc:
            raise SolverRuntimeError("worker_start_failed", metadata=metadata) from exc
        finally:
            had_error = sys.exc_info()[0] is not None
            container_removed = True
            if container is not None:
                # Killing an attached CLI does not stop its daemon-side worker.
                # Cleanup owns exactly one random name, never all containers.
                try:
                    container_removed = container.cleanup()
                except Exception:
                    container_removed = False
                metadata["container_cleanup"] = "confirmed" if container_removed else "unverified"
                if not container_removed:
                    # Releasing local quota must not enable an endless series
                    # of new containers while previous ones may still be alive.
                    _CONTAINER_RECOVERY_REQUIRED.set()
            # Also run on KeyboardInterrupt. No child is allowed to survive its request.
            if job is not None:
                job.close()
            if process is not None:
                if os.name != "nt":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if process.poll() is None:
                    process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # A broken child must not strand the request thread or the
                    # semaphore slot.  Retry the hard kill once, then continue
                    # cleanup; the result is already non-publishable.
                    try:
                        process.kill()
                    except OSError:
                        pass
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        metadata["cleanup_warning"] = "worker_reap_timeout"
                if writer is not None:
                    writer.join(timeout=1)
                for reader in readers:
                    reader.thread.join(timeout=1)
                for pipe in (process.stdin, process.stdout, process.stderr):
                    pipe.close()
            if not container_removed and not had_error:
                raise SolverRuntimeError("container_cleanup_failed", metadata=metadata)
