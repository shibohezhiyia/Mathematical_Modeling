"""Opt-in real-daemon smoke checks, never substitutes mocks for deployment.

Requires a built, pinned image and deployment-owned v2 attestation. Does not
start Docker Desktop, pull images, or alter the host daemon configuration.
"""

import json
import math
import os

import pytest

from core.os_sandbox import ContainerWorkerSession
from core.solver_runtime import SolverLimits, SolverProcessRunner


pytestmark = pytest.mark.skipif(
    os.environ.get("MATHMODEL_TEST_REAL_SANDBOX") != "1",
    reason="requires explicitly enabled real container runtime and attested worker image",
)


def test_real_container_configuration_and_reclamation():
    session = ContainerWorkerSession(memory_mb=512, disk_bytes=16 * 1024**2)
    try:
        session.prepare(30)
        inspected = session._call([session.runtime, "inspect", session.name], 10)
        assert inspected.returncode == 0
        document = json.loads(inspected.stdout)[0]
        host = document["HostConfig"]
        assert document["Config"]["User"] == "65532:65532"
        assert document["Config"]["OpenStdin"] is True
        assert host["ReadonlyRootfs"] is True
        assert host["NetworkMode"] == "none"
        assert host["Privileged"] is False
        assert host["Memory"] == host["MemorySwap"] == 512 * 1024**2
        assert host["PidsLimit"] == 64
        assert "ALL" in [cap.upper() for cap in host["CapDrop"]]
        assert any("no-new-privileges" in flag for flag in host["SecurityOpt"])
        assert not host.get("Binds")
        assert all(mount["Type"] == "tmpfs" for mount in document.get("Mounts", []))
        assert "size=16777216" in host["Tmpfs"]["/tmp"]
    finally:
        assert session.cleanup(), f"cannot confirm cleanup of {session.name}"


def test_real_container_runs_numeric_contract_and_cleans_up():
    result = SolverProcessRunner().execute(
        "linear_ode/v1",
        {"matrix": [[-1.0]], "initial": [1.0], "times": [0.0, 1.0]},
        limits=SolverLimits(isolation_mode="strict_os", memory_mb=512, wall_seconds=60),
    )
    assert result["states"][-1][0] == pytest.approx(math.exp(-1), rel=1e-6)
    assert result["execution_supervision"]["container_cleanup"] == "confirmed"
    assert result["execution_supervision"]["os_supervised"] is True
