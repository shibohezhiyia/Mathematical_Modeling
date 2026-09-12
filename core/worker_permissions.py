"""Defense-in-depth permission policy for numerical workers.

Resource limits do not restrict files or sockets.  This module provides a
small, auditable policy evaluator and Python audit-hook adapter.  It is not a
replacement for a container, Windows restricted token, or Linux namespace;
callers must deploy a stronger OS boundary for hostile code.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Iterable


class WorkerPermissionError(RuntimeError):
    pass


def _root(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser().resolve()
    if str(candidate).rstrip("\\/") == str(candidate.anchor).rstrip("\\/"):
        raise WorkerPermissionError("broad_filesystem_root_not_allowed")
    return candidate


@dataclass(frozen=True)
class WorkerPermissionPolicy:
    allowed_roots: tuple[str, ...]
    allow_network: bool = False
    allow_subprocess: bool = False
    max_write_bytes: int = 64 * 1024 * 1024

    def validate(self) -> "WorkerPermissionPolicy":
        if not isinstance(self.allowed_roots, tuple) or not self.allowed_roots:
            raise WorkerPermissionError("allowed_roots_required")
        roots = tuple(str(_root(item)) for item in self.allowed_roots)
        if type(self.allow_network) is not bool or type(self.allow_subprocess) is not bool:
            raise WorkerPermissionError("permission_flags_must_be_boolean")
        if type(self.max_write_bytes) is not int or not 1 <= self.max_write_bytes <= 4 * 1024**3:
            raise WorkerPermissionError("invalid_write_quota")
        return WorkerPermissionPolicy(roots, self.allow_network, self.allow_subprocess, self.max_write_bytes)

    def public(self) -> dict[str, Any]:
        config = self.validate()
        return {"allowed_roots": list(config.allowed_roots), "allow_network": config.allow_network,
                "allow_subprocess": config.allow_subprocess, "max_write_bytes": config.max_write_bytes,
                "policy": "python_audit_defense_in_depth_not_os_sandbox"}


def _inside(path: str | os.PathLike[str], roots: Iterable[str]) -> bool:
    try:
        candidate = Path(path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    return any(candidate == Path(root) or Path(root) in candidate.parents for root in roots)


def directory_usage_bytes(root: str | os.PathLike[str], *, max_entries: int = 100_000) -> int:
    """Return a bounded regular-file byte count for a worker root.

    Symlinks are not followed.  A scan is deliberately conservative: if the
    entry budget is exceeded the quota check fails closed rather than
    claiming that the directory is small enough.
    """
    base = _root(root)
    if not base.exists() or not base.is_dir():
        raise WorkerPermissionError("quota_root_not_directory")
    if type(max_entries) is not int or max_entries < 1:
        raise WorkerPermissionError("invalid_quota_scan_budget")
    total = 0
    seen = 0
    try:
        for path in base.rglob("*"):
            seen += 1
            if seen > max_entries:
                raise WorkerPermissionError("quota_scan_budget_exceeded")
            if path.is_symlink():
                continue
            if path.is_file():
                total += int(path.stat().st_size)
    except (OSError, RuntimeError) as exc:
        raise WorkerPermissionError("quota_scan_failed") from exc
    return total


def enforce_write_quota(policy: WorkerPermissionPolicy, root: str | os.PathLike[str] | None = None) -> dict[str, int]:
    """Check current usage against ``max_write_bytes`` and return evidence."""
    config = policy.validate()
    selected = str(_root(root or config.allowed_roots[0]))
    # Callers may pass the destination file from an ``open`` audit event.
    # Normalize it to the nearest existing directory before scanning.
    candidate_path = Path(selected)
    if not candidate_path.is_dir():
        candidate_path = candidate_path.parent
        selected = str(_root(candidate_path))
    if not _inside(selected, config.allowed_roots) and selected not in config.allowed_roots:
        raise WorkerPermissionError("quota_root_outside_allowed_roots")
    used = directory_usage_bytes(selected)
    if used > config.max_write_bytes:
        raise WorkerPermissionError("write_quota_exceeded")
    return {"root": selected, "used_bytes": used, "max_write_bytes": config.max_write_bytes,
            "remaining_bytes": config.max_write_bytes - used}


def check_audit_event(policy: WorkerPermissionPolicy, event: str, args: tuple[Any, ...] = ()) -> None:
    """Raise a stable error if a Python audit event violates the policy."""
    config = policy.validate()
    # Deny socket creation as well as DNS/connect/bind. Blocking only the
    # latter still lets untrusted code allocate network handles first.
    if event in {"socket.__new__", "socket.connect", "socket.bind", "socket.getaddrinfo",
                 "socket.gethostbyname", "socket.gethostbyname_ex", "socket.gethostbyaddr"} and not config.allow_network:
        raise WorkerPermissionError("network_access_denied")
    if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn"} and not config.allow_subprocess:
        raise WorkerPermissionError("subprocess_access_denied")
    path_events = {
        "open", "os.open", "shutil.copyfile", "shutil.move", "shutil.copy",
        "shutil.copy2", "shutil.rmtree", "os.listdir", "os.scandir", "os.remove",
        "os.unlink", "os.mkdir", "os.rmdir", "os.chmod", "os.chown",
        "os.rename", "os.replace", "os.link", "os.symlink",
    }
    if event in path_events:
        # Rename/link operations have both a source and destination.  Check
        # every path so a worker cannot smuggle data from an allowed temp
        # directory to an untrusted location (or vice versa).
        paths = list(args[:2]) if event in {"os.rename", "os.replace", "os.link", "os.symlink"} else list(args[:1])
        if not paths:
            raise WorkerPermissionError("filesystem_path_denied")
        for path in paths:
            _check_path(config, path)
        # Audit hooks cannot know the eventual write size, but refusing a new
        # write once the current root is full prevents unbounded growth.  The
        # parent/worker should call ``enforce_write_quota`` after the job too.
        if event in {"open", "os.open"} and len(args) > 1:
            mode = args[1]
            flags = args[2] if len(args) > 2 else 0
            writing = (isinstance(mode, str) and any(token in mode for token in ("w", "a", "+", "x"))) or (
                isinstance(flags, int) and not isinstance(flags, bool) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
            )
            exempt = isinstance(paths[0], int) or (
                isinstance(paths[0], (str, os.PathLike))
                and os.path.abspath(os.fspath(paths[0])) == os.path.abspath(os.devnull)
            )
            if writing and not exempt:
                quota = enforce_write_quota(config, paths[0])
                if quota["remaining_bytes"] <= 0:
                    raise WorkerPermissionError("write_quota_exceeded")


def _check_path(config: WorkerPermissionPolicy, path: Any) -> None:
    # An integer file descriptor is already owned by this worker; there is no
    # path to resolve. New path-based opens still go through the root check.
    if isinstance(path, int) and not isinstance(path, bool):
        return
    if isinstance(path, (str, os.PathLike)) and os.path.abspath(os.fspath(path)) == os.path.abspath(os.devnull):
        return
    if not isinstance(path, (str, os.PathLike)) or not _inside(path, config.allowed_roots):
        raise WorkerPermissionError("filesystem_path_denied")


def make_audit_hook(policy: WorkerPermissionPolicy):
    """Return an audit hook suitable for ``sys.addaudithook``."""
    config = policy.validate()

    def hook(event: str, args: tuple[Any, ...]) -> None:
        check_audit_event(config, event, args)

    return hook


__all__ = ["WorkerPermissionError", "WorkerPermissionPolicy", "check_audit_event", "make_audit_hook",
           "directory_usage_bytes", "enforce_write_quota"]
