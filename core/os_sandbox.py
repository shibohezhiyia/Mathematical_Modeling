"""Fail-closed OS isolation capability detection.

Python audit hooks and resource limits are defense-in-depth, not an OS
boundary.  This module makes that distinction executable: callers requesting
``strict`` isolation must obtain a configured container capability first, or the
run is refused instead of silently falling back to the host process.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
import shutil
import subprocess
import math
import re
import secrets
import time
from typing import Any


class OSSandboxError(RuntimeError):
    pass


class OSSandboxCancelled(OSSandboxError):
    pass


def _load_attestation(runtime: str, image: str) -> dict[str, Any]:
    """Load a deployment-owned isolation statement and fail closed.

    An environment flag is too easy to set accidentally (or maliciously), so
    strict mode also requires a small JSON statement produced by the process
    supervisor.  This is still an attestation input, not a cryptographic
    proof; the deployment must protect the file and record its provenance.
    """
    raw_path = os.environ.get("MATHMODEL_OS_SANDBOX_ATTESTATION_FILE", "").strip()
    if not raw_path:
        raise OSSandboxError("os_sandbox_attestation_file_required")
    path = Path(raw_path).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise OSSandboxError("os_sandbox_attestation_file_missing")
    try:
        if path.stat().st_size > 64 * 1024:
            raise OSSandboxError("os_sandbox_attestation_file_too_large")
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSSandboxError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OSSandboxError("os_sandbox_attestation_file_invalid") from exc
    if not isinstance(document, dict) or document.get("schema_version") != "mathmodel.os-attestation/v2":
        raise OSSandboxError("os_sandbox_attestation_schema_invalid")
    if document.get("runtime") != runtime or document.get("image") != image:
        raise OSSandboxError("os_sandbox_attestation_runtime_mismatch")
    required = {
        "network_none": True,
        "read_only_root": True,
        "non_root": True,
        "pids_limit": 64,
        "no_host_mounts": True,
    }
    for key, expected in required.items():
        if type(document.get(key)) is not type(expected) or document.get(key) != expected:
            raise OSSandboxError(f"os_sandbox_attestation_{key}_missing")
    return document


def detect_os_sandbox() -> dict[str, Any]:
    docker = shutil.which("docker")
    unshare = shutil.which("unshare") if os.name != "nt" else None
    podman = shutil.which("podman")
    runtimes = []
    if docker:
        runtimes.append("docker")
    if podman:
        runtimes.append("podman")
    if unshare:
        runtimes.append("unshare")
    return {
        "schema_version": "mathmodel.os-sandbox/v1",
        "platform": os.name,
        "runtimes": runtimes,
        "strict_available": bool(docker or podman),
        "availability_scope": "CLI_discovery_only; daemon_and_image_not_tested",
        "python_layer": "audit_hook_and_trusted_directory_only",
        "policy": "strict_mode_fails_closed_when_no_os_runtime_is_available",
    }


def require_strict_os_sandbox() -> dict[str, Any]:
    capabilities = detect_os_sandbox()
    if not capabilities["strict_available"]:
        raise OSSandboxError("os_sandbox_runtime_unavailable")
    # Discovery alone is never an attestation.  Deployment must set this only
    # after its external supervisor/container has actually been installed and
    # recorded; a normal in-process web server therefore fails closed.
    runtime = os.environ.get("MATHMODEL_OS_SANDBOX_RUNTIME", "").strip().lower()
    image = os.environ.get("MATHMODEL_OS_SANDBOX_IMAGE", "").strip()
    if os.environ.get("MATHMODEL_OS_SANDBOX_ATTESTED") != "1":
        raise OSSandboxError("os_sandbox_attestation_required")
    if runtime not in {"docker", "podman"} or not image or not shutil.which(runtime):
        raise OSSandboxError("os_sandbox_supervisor_configuration_missing")
    if not re.fullmatch(r"(?:[A-Za-z0-9][A-Za-z0-9._:/-]*@)?sha256:[a-f0-9]{64}", image):
        raise OSSandboxError("os_sandbox_immutable_image_required")
    attestation = _load_attestation(runtime, image)
    capabilities["runtime"] = runtime
    capabilities["image"] = image
    capabilities["attestation"] = {
        "schema_version": attestation["schema_version"],
        "path": os.environ.get("MATHMODEL_OS_SANDBOX_ATTESTATION_FILE", ""),
        "network_none": True,
        "read_only_root": True,
        "non_root": True,
        "pids_limit": 64,
        "no_host_mounts": True,
    }
    return capabilities


def build_container_command(runtime: str, image: str, command: list[str], *, memory_mb: int, cpu_limit: float = 1.0,
                            disk_bytes: int = 64 * 1024 * 1024) -> list[str]:
    """Build a no-network, read-only-root command for an external supervisor.

    Only trusted deployment configuration may supply the image. No host
    directory is mounted; stdin/stdout carry the bounded numeric contract.
    """
    if (runtime not in {"docker", "podman"} or not isinstance(image, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@-]{0,255}", image)
            or not isinstance(command, list) or not command
            or any(type(arg) is not str or "\x00" in arg for arg in command)):
        raise OSSandboxError("container_command_contract_invalid")
    if type(memory_mb) is not int or not 64 <= memory_mb <= 32768:
        raise OSSandboxError("container_memory_invalid")
    if type(cpu_limit) not in (int, float) or not math.isfinite(cpu_limit) or not 0 < float(cpu_limit) <= 64:
        raise OSSandboxError("container_cpu_invalid")
    if type(disk_bytes) is not int or not 1024 * 1024 <= disk_bytes <= 4 * 1024**3:
        raise OSSandboxError("container_disk_invalid")
    return [runtime, "run", "--rm", "--interactive", "--pull", "never",
            "--network", "none", "--read-only", "--pids-limit", "64",
            "--user", "65532:65532", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--no-healthcheck", "--log-driver", "none",
            "--memory", f"{memory_mb}m", "--memory-swap", f"{memory_mb}m",
            "--cpus", str(float(cpu_limit)), "--ipc", "none", "--ulimit", "core=0:0",
            "--tmpfs", f"/tmp:rw,noexec,nosuid,nodev,size={disk_bytes},mode=1777",
            "--workdir", "/tmp", "--entrypoint", "/usr/bin/env", image,
            "-i", "PATH=/usr/local/bin:/usr/bin:/bin", "HOME=/tmp", "TMPDIR=/tmp",
            "LANG=C.UTF-8", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
            "MKL_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1", "VECLIB_MAXIMUM_THREADS=1", *command]


def build_attested_worker_command(directory: str, *, memory_mb: int, cpu_limit: float = 1.0,
                                  disk_bytes: int = 64 * 1024 * 1024) -> list[str]:
    """Build the actual worker command for a configured container supervisor."""
    capabilities = require_strict_os_sandbox()
    runtime, image = capabilities["runtime"], capabilities["image"]
    # directory remains in the signature for compatibility; it never crosses
    # the container boundary. Code is baked into the reviewed worker image.
    return build_container_command(runtime, image,
        ["python", "-I", "-B", "-u", "/app/core/solver_worker.py"], memory_mb=memory_mb,
        cpu_limit=cpu_limit, disk_bytes=disk_bytes)


def supervisor_environment() -> dict[str, str]:
    """Host CLI configuration only; never passed as container environment."""
    allowed = {"SYSTEMROOT", "WINDIR", "PATH", "HOME", "USERPROFILE", "DOCKER_CONFIG",
               "DOCKER_CONTEXT", "DOCKER_HOST", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH",
               "XDG_RUNTIME_DIR", "XDG_CONFIG_HOME", "CONTAINER_HOST", "CONTAINER_CONNECTION"}
    return {key: value for key, value in os.environ.items() if key.upper() in allowed}


class ContainerWorkerSession:
    """Create first, then start/attach; reclaim the container separately from CLI."""

    def __init__(self, *, memory_mb: int, disk_bytes: int):
        capabilities = require_strict_os_sandbox()
        self.image = capabilities["image"]
        self.runtime = shutil.which(capabilities["runtime"])
        if not self.runtime:
            raise OSSandboxError("os_sandbox_runtime_unavailable")
        self.environment = supervisor_environment()
        self.name = "mathmodel-worker-" + secrets.token_hex(16)
        base = build_container_command(capabilities["runtime"], self.image,
            ["python", "-I", "-B", "-u", "/app/core/solver_worker.py"],
            memory_mb=memory_mb, disk_bytes=disk_bytes)
        self.create_command = [self.runtime, "create", "--name", self.name, *base[3:]]
        self.attempted = False
        self.creation_uncertain = False

    def _call(self, args: list[str], timeout: float, *, cancel=None):
        deadline = time.monotonic() + timeout
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=self.environment, shell=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    raise OSSandboxCancelled("os_sandbox_cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(args, timeout)
                try:
                    output, _ = process.communicate(timeout=min(0.1, remaining))
                    return subprocess.CompletedProcess(args, process.returncode, stdout=output)
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=1)
            finally:
                process.stdout.close()

    def prepare(self, timeout: float, *, cancel=None) -> list[str]:
        deadline = time.monotonic() + timeout
        # Image-declared VOLUMEs could introduce unbounded writable storage.
        inspected = self._call([self.runtime, "image", "inspect", "--format", "{{json .Config.Volumes}}", self.image], timeout, cancel=cancel)
        if inspected.returncode != 0 or inspected.stdout.strip() not in (b"null", b"{}"):
            raise OSSandboxError("os_sandbox_image_missing_or_declares_volumes")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(self.create_command, timeout)
        self.attempted = True
        self.creation_uncertain = True
        created = self._call(self.create_command, remaining, cancel=cancel)
        self.creation_uncertain = False
        if created.returncode != 0:
            raise OSSandboxError("os_sandbox_container_create_failed")
        return [self.runtime, "start", "--attach", "--interactive", self.name]

    def cleanup(self) -> bool:
        if not self.attempted:
            return True
        try:
            removed = self._call([self.runtime, "rm", "--force", "--volumes", self.name], 5)
            if removed.returncode == 0:
                return True
            # A nonzero removal alone cannot distinguish absent from daemon
            # failure; verify absence without parsing localized error strings.
            remaining = self._call([self.runtime, "ps", "--all", "--filter",
                f"name=^/{self.name}$", "--format", "{{.Names}}"], 5)
            # If create was interrupted, the daemon may finish it later.
            # A currently absent name cannot prove that request was aborted.
            return not self.creation_uncertain and remaining.returncode == 0 and not remaining.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return False


__all__ = ["OSSandboxError", "detect_os_sandbox", "require_strict_os_sandbox",
           "build_container_command", "build_attested_worker_command", "ContainerWorkerSession"]
