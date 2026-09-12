"""OS resource limits for trusted numerical workers, NOT a permission sandbox.

Windows jobs cap committed memory and process count. Linux RLIMIT_AS caps
address space (not RSS); a process group permits timeout cleanup. Neither
mechanism denies reading files or using the network.
"""

from __future__ import annotations

import os
import sys


class ResourceIsolationUnavailable(RuntimeError):
    pass


class WindowsJob:
    """Attach a worker BEFORE sending input; closing the job kills its processes."""

    def __init__(self, memory_bytes: int) -> None:
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self._api = ctypes.WinDLL("kernel32", use_last_error=True)
        self._api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self._api.CreateJobObjectW.restype = wintypes.HANDLE
        self._api.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ]
        self._api.SetInformationJobObject.restype = wintypes.BOOL
        self._api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self._api.AssignProcessToJobObject.restype = wintypes.BOOL
        self._api.CloseHandle.argtypes = [wintypes.HANDLE]
        self._api.CloseHandle.restype = wintypes.BOOL
        self._handle = self._api.CreateJobObjectW(None, None)
        if not self._handle:
            raise ResourceIsolationUnavailable("cannot create a Windows resource job")
        limits = ExtendedLimits()
        # ACTIVE_PROCESS | JOB_MEMORY | KILL_ON_JOB_CLOSE. No breakaway flag.
        limits.BasicLimitInformation.LimitFlags = 0x8 | 0x200 | 0x2000
        limits.BasicLimitInformation.ActiveProcessLimit = 1
        limits.JobMemoryLimit = memory_bytes
        if not self._api.SetInformationJobObject(
            self._handle, 9, ctypes.byref(limits), ctypes.sizeof(limits),
        ):
            self.close()
            raise ResourceIsolationUnavailable("cannot install Windows job limits")

    def attach(self, process: object) -> None:
        if not self._api.AssignProcessToJobObject(self._handle, int(process._handle)):
            raise ResourceIsolationUnavailable("cannot assign worker to Windows resource job")

    def close(self) -> None:
        if self._handle:
            self._api.CloseHandle(self._handle)
            self._handle = None


def install_linux_limits(memory_bytes: int) -> None:
    """Called in the fresh worker before numerical imports, never in the server."""
    if sys.platform != "linux":
        raise ResourceIsolationUnavailable("this platform has no tested resource backend")
    import resource

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        ceiling = memory_bytes if hard == resource.RLIM_INFINITY else min(memory_bytes, hard)
        resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (OSError, ValueError) as exc:
        raise ResourceIsolationUnavailable("cannot install Linux resource limits") from exc


def resource_backend() -> str:
    if os.name == "nt":
        return "windows_job_committed_memory"
    if sys.platform == "linux":
        return "linux_rlimit_address_space"
    raise ResourceIsolationUnavailable("this platform has no implemented resource backend")
