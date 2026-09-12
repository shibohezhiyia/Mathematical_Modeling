"""Trusted test-only process failures. This is NOT reachable by the solver API."""

import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.solver_process_limits import install_linux_limits


request = json.loads(sys.stdin.buffer.read())
if os.name != "nt":
    install_linux_limits(request["limits"]["memory_mb"] * 1024 * 1024)
mode = sys.argv[1]
if mode == "hang":
    while True:
        time.sleep(0.1)
elif mode == "flood":
    while True:
        sys.stdout.buffer.write(b"x" * 8192)
        sys.stdout.buffer.flush()
elif mode == "stderr_flood":
    while True:
        sys.stderr.buffer.write(b"x" * 8192)
        sys.stderr.buffer.flush()
elif mode == "crash":
    sys.stderr.write("PRIVATE_TEST_TRACEBACK_DO_NOT_EXPOSE")
    sys.exit(7)
elif mode == "memory":
    blocks = []
    try:
        # Bounded adversary: never allocate above 192 MiB even if OS isolation
        # were broken. Test uses a 128 MiB job/address-space limit.
        for _ in range(24):
            blocks.append(bytearray(8 * 1024 * 1024))
    except MemoryError:
        blocks.clear()
        response = {"status": "failed", "code": "memory_limit"}
    else:
        response = {"status": "ok", "result": {"limit_not_enforced": True}}
elif mode == "invalid":
    sys.stdout.write('{"protocol":"wrong","status":"ok","result":{}}')
    sys.exit(0)
elif mode == "nonfinite":
    sys.stdout.write('{"protocol":"mathmodel.solver-process/v1","status":"ok","result":{"value":NaN}}')
    sys.exit(0)
elif mode == "disk":
    Path.cwd().joinpath("oversized-intermediate.bin").write_bytes(b"x" * (2 * 1024 * 1024))
    response = {"status": "ok", "result": {"disk_write": True}}
else:
    response = {"status": "ok", "result": {
        "secret_inherited": "TEST_SOLVER_API_KEY" in os.environ,
        "pythonpath_inherited": "PYTHONPATH" in os.environ,
        "threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "temporary_directory": Path.cwd().name.startswith("mathmodel-solver-"),
        "payload_size": len(request["contract"].get("text", "")),
    }}
response["protocol"] = "mathmodel.solver-process/v1"
sys.stdout.write(json.dumps(response))
