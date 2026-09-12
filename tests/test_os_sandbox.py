import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
import pytest

from core import os_sandbox as sandbox
from core import solver_runtime as runtime
from core.os_sandbox import OSSandboxError, build_attested_worker_command, build_container_command, detect_os_sandbox, require_strict_os_sandbox
from core.solver_runtime import SolverLimits, SolverProcessRunner, SolverRuntimeError


@pytest.fixture(autouse=True)
def isolated_recovery_latch(monkeypatch):
    monkeypatch.setattr(runtime, "_CONTAINER_RECOVERY_REQUIRED", threading.Event())


def test_os_sandbox_detection_is_explicit_and_container_command_is_hardened():
    report = detect_os_sandbox()
    assert report["python_layer"] == "audit_hook_and_trusted_directory_only"
    command = build_container_command("docker", "python:3.11-slim", ["python", "-m", "worker"], memory_mb=256)
    assert "--network" in command and "none" in command and "--read-only" in command


def test_strict_mode_never_silently_falls_back_to_python_worker(monkeypatch):
    monkeypatch.delenv("MATHMODEL_OS_SANDBOX_ATTESTED", raising=False)
    with pytest.raises((OSSandboxError, SolverRuntimeError)):
        require_strict_os_sandbox()
    with pytest.raises(SolverRuntimeError, match="permission_isolation_unavailable"):
        SolverProcessRunner().execute("linear_ode/v1", {}, limits=SolverLimits(isolation_mode="strict_os"))


@pytest.fixture
def attested(monkeypatch, tmp_path):
    monkeypatch.setenv("MATHMODEL_OS_SANDBOX_ATTESTED", "1")
    monkeypatch.setenv("MATHMODEL_OS_SANDBOX_RUNTIME", "docker")
    image = "sha256:" + "a" * 64
    monkeypatch.setenv("MATHMODEL_OS_SANDBOX_IMAGE", image)
    attestation = tmp_path / "os-attestation.json"
    attestation.write_text(json.dumps({
        "schema_version": "mathmodel.os-attestation/v2",
        "runtime": "docker", "image": image,
        "network_none": True, "read_only_root": True,
        "non_root": True, "pids_limit": 64, "no_host_mounts": True,
    }), encoding="utf-8")
    monkeypatch.setenv("MATHMODEL_OS_SANDBOX_ATTESTATION_FILE", str(attestation))
    monkeypatch.setattr("core.os_sandbox.shutil.which", lambda name: "C:/docker.exe" if name == "docker" else None)
    return attestation, image


def test_attested_command_orders_flags_and_never_mounts_host(attested, tmp_path, monkeypatch):
    _, image = attested
    monkeypatch.setenv("TEST_SOLVER_API_KEY", "test-only-secret")
    command = build_attested_worker_command(str(tmp_path), memory_mb=256, disk_bytes=2**20)
    options = command[:command.index(image)]
    for option in ("--interactive", "--read-only", "--cap-drop", "--user", "--tmpfs", "--entrypoint"):
        assert option in options
    assert options[options.index("--user") + 1] == "65532:65532"
    assert options[options.index("--pull") + 1] == "never"
    assert "size=1048576" in options[options.index("--tmpfs") + 1]
    assert "--volume" not in command and "--mount" not in command and "-v" not in command
    assert str(tmp_path) not in " ".join(command)
    assert "test-only-secret" not in " ".join(command)
    assert command[command.index(image) + 1] == "-i"
    assert command[-1] == "/app/core/solver_worker.py"


@pytest.mark.parametrize("changes", [
    {"schema_version": "mathmodel.os-attestation/v1"}, {"network_none": 1},
    {"pids_limit": True}, {"non_root": False}, {"no_host_mounts": False},
])
def test_attestation_rejects_stale_or_false_policy(attested, changes):
    path, _ = attested
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(OSSandboxError):
        require_strict_os_sandbox()


def test_mutable_image_is_not_accepted(attested, monkeypatch):
    monkeypatch.setenv("MATHMODEL_OS_SANDBOX_IMAGE", "worker:latest")
    with pytest.raises(OSSandboxError, match="immutable_image"):
        require_strict_os_sandbox()


def test_unshare_discovery_does_not_enable_unimplemented_backend(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/bin/unshare" if name == "unshare" else None)
    assert detect_os_sandbox()["strict_available"] is False


@pytest.mark.parametrize("kw", [
    {"cpu_limit": float("nan")}, {"cpu_limit": True}, {"memory_mb": True},
    {"disk_bytes": 1}, {"image": "--privileged"}, {"command": ["python\x00"]},
])
def test_command_rejects_malformed_configuration(kw):
    args = dict(runtime="docker", image="worker:local", command=["python"], memory_mb=256)
    args.update(kw)
    with pytest.raises(OSSandboxError):
        build_container_command(**args)


def test_supervisor_environment_retains_context_without_app_secrets(monkeypatch):
    monkeypatch.setenv("DOCKER_CONTEXT", "desktop-linux")
    monkeypatch.setenv("TEST_SOLVER_API_KEY", "test-only")
    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    env = sandbox.supervisor_environment()
    assert env["DOCKER_CONTEXT"] == "desktop-linux"
    assert "TEST_SOLVER_API_KEY" not in env and "PYTHONPATH" not in env


def test_session_creates_before_attach_and_cleans_its_exact_name(attested, monkeypatch):
    calls = []
    session = sandbox.ContainerWorkerSession(memory_mb=256, disk_bytes=2**20)
    def call(args, timeout, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=b"null" if "inspect" in args else b"")
    monkeypatch.setattr(session, "_call", call)
    command = session.prepare(10)
    assert calls[0][1:3] == ["image", "inspect"]
    assert calls[1][1:4] == ["create", "--name", session.name]
    assert "--rm" not in calls[1]
    assert command[1:] == ["start", "--attach", "--interactive", session.name]
    assert session.cleanup()
    assert calls[-1][1:] == ["rm", "--force", "--volumes", session.name]


def test_image_declared_volumes_are_rejected_before_creation(attested, monkeypatch):
    session = sandbox.ContainerWorkerSession(memory_mb=256, disk_bytes=2**20)
    monkeypatch.setattr(session, "_call", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=b'{"/data":{}}'))
    with pytest.raises(OSSandboxError, match="declares_volumes"):
        session.prepare(10)
    assert not session.attempted
    assert session.cleanup()


@pytest.mark.parametrize("exit_code,output,expected", [(0, b"", True), (1, b"", False), (0, b"still-present", False)])
def test_failed_removal_requires_confirmed_absence(attested, monkeypatch, exit_code, output, expected):
    session = sandbox.ContainerWorkerSession(memory_mb=256, disk_bytes=2**20)
    session.attempted = True
    def call(args, timeout):
        return SimpleNamespace(returncode=1 if args[1] == "rm" else exit_code, stdout=output)
    monkeypatch.setattr(session, "_call", call)
    assert session.cleanup() is expected


@pytest.mark.parametrize("mode,expected", [
    ("ok", None), ("hang", "timeout"), ("crash", "worker_exit"),
    ("invalid", "invalid_response"), ("flood", "output_limit"),
    ("cancel", "cancelled"), ("create_timeout", "timeout"),
    ("create_error", "permission_isolation_unavailable"), ("start_error", "worker_start_failed"),
])
def test_runner_container_lifecycle_without_daemon(monkeypatch, mode, expected):
    # A real bounded probe process tests runner wiring, NOT container isolation.
    sessions, processes = [], []
    cancel = threading.Event()
    class Session:
        image = "sha256:" + "b" * 64
        name = "mathmodel-worker-test"
        environment = runtime._worker_environment(".")
        def __init__(self, **kw):
            self.cleaned = False
            sessions.append(self)
        def prepare(self, timeout, **kwargs):
            if mode == "create_timeout":
                raise subprocess.TimeoutExpired("create", timeout)
            if mode == "create_error":
                raise OSSandboxError("create failed")
            if mode == "cancel":
                cancel.set()
            if mode == "start_error":
                return [str(Path(__file__).parent / "nonexistent-runtime.exe")]
            probe = Path(__file__).parent / "fixtures" / "solver_worker_probe.py"
            return [sys.executable, "-I", "-B", "-u", str(probe), mode]
        def cleanup(self):
            self.cleaned = True
            return True
    monkeypatch.setattr(sandbox, "require_strict_os_sandbox", lambda: {})
    monkeypatch.setattr(sandbox, "ContainerWorkerSession", Session)
    def unexpected(*args):
        pytest.fail("strict mode must not use the native process memory backend")
    monkeypatch.setattr(runtime, "WindowsJob", unexpected)
    monkeypatch.setattr(runtime, "resource_backend", unexpected)
    popen = runtime.subprocess.Popen
    def track(*args, **kw):
        process = popen(*args, **kw)
        processes.append(process)
        return process
    monkeypatch.setattr(runtime.subprocess, "Popen", track)
    limits = SolverLimits(isolation_mode="strict_os", wall_seconds=1 if mode == "hang" else 10)
    runner = SolverProcessRunner()
    if expected:
        with pytest.raises(SolverRuntimeError, match=expected):
            runner.execute("linear_ode/v1", {}, limits=limits, cancel=cancel)
    else:
        result = runner.execute("linear_ode/v1", {}, limits=limits)
        assert result["execution_supervision"]["container_cleanup"] == "confirmed"
        assert result["execution_supervision"]["memory_backend"] == "container_cgroup"
    assert sessions and all(session.cleaned for session in sessions)
    assert all(process.poll() is not None for process in processes)


@pytest.mark.parametrize("failed_run", [False, True])
def test_unconfirmed_cleanup_never_returns_success(monkeypatch, failed_run):
    class Session:
        image, name, environment = "test-image", "test-container", {}
        def __init__(self, **kwargs):
            pass
        def prepare(self, timeout, **kwargs):
            if failed_run:
                raise OSSandboxError("create failed")
            return [sys.executable, "-c", 'import sys; sys.stdin.read(); print(\'{"protocol":"mathmodel.solver-process/v1","status":"ok","result":{}}\')']
        def cleanup(self):
            return False
    monkeypatch.setattr(sandbox, "require_strict_os_sandbox", lambda: {})
    monkeypatch.setattr(sandbox, "ContainerWorkerSession", Session)
    expected = "permission_isolation_unavailable" if failed_run else "container_cleanup_failed"
    with pytest.raises(SolverRuntimeError, match=expected) as err:
        SolverProcessRunner().execute("linear_ode/v1", {}, limits=SolverLimits(isolation_mode="strict_os"))
    assert err.value.metadata["container_cleanup"] == "unverified"
    with pytest.raises(SolverRuntimeError, match="container_recovery_required"):
        SolverProcessRunner().execute("linear_ode/v1", {}, limits=SolverLimits(isolation_mode="strict_os"))


def test_interrupted_create_cannot_claim_absence_is_final(attested, monkeypatch):
    session = sandbox.ContainerWorkerSession(memory_mb=256, disk_bytes=2**20)
    def call(args, timeout, **kwargs):
        if args[1] == "create":
            raise subprocess.TimeoutExpired(args, timeout)
        return SimpleNamespace(returncode=1 if args[1] == "rm" else 0,
                               stdout=b"null" if "inspect" in args else b"")
    monkeypatch.setattr(session, "_call", call)
    with pytest.raises(subprocess.TimeoutExpired):
        session.prepare(2)
    assert session.creation_uncertain
    assert not session.cleanup()


def test_cli_control_operation_responds_to_cancellation(attested, monkeypatch):
    session = sandbox.ContainerWorkerSession(memory_mb=256, disk_bytes=2**20)
    cancel = threading.Event()
    processes = []
    popen = subprocess.Popen
    def track(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        cancel.set()
        return process
    monkeypatch.setattr(sandbox.subprocess, "Popen", track)
    with pytest.raises(sandbox.OSSandboxCancelled):
        session._call([sys.executable, "-c", "import time; time.sleep(30)"], 60, cancel=cancel)
    assert all(process.poll() is not None for process in processes)
