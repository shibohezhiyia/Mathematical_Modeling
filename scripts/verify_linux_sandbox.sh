#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. This script deliberately does not enable the
# application: it builds and records the real worker image evidence first.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${MATHMODEL_OS_SANDBOX_RUNTIME:-docker}"
TAG="${MATHMODEL_OS_SANDBOX_TAG:-mathmodel-worker:local}"
OUT="${MATHMODEL_SANDBOX_EVIDENCE_DIR:-${ROOT}/data/runtime/linux-sandbox}"
mkdir -p "$OUT"

command -v "$RUNTIME" >/dev/null || { echo "missing runtime: $RUNTIME" >&2; exit 2; }
"$RUNTIME" build --pull=never -f "$ROOT/deploy/solver/Dockerfile" -t "$TAG" "$ROOT"
IMAGE="$($RUNTIME image inspect --format '{{.Id}}' "$TAG")"
[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]] || { echo "invalid image digest: $IMAGE" >&2; exit 3; }

python3 - "$OUT/attestation.json" "$RUNTIME" "$IMAGE" <<'PY'
import json, sys
path, runtime, image = sys.argv[1:]
payload = {
    "schema_version": "mathmodel.os-attestation/v2", "runtime": runtime,
    "image": image, "network_none": True, "read_only_root": True,
    "non_root": True, "pids_limit": 64, "no_host_mounts": True,
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
PY

"$RUNTIME" image inspect "$IMAGE" > "$OUT/image-inspect.json"
"$RUNTIME" version > "$OUT/runtime-version.txt"
echo "image=$IMAGE"
echo "attestation=$OUT/attestation.json"

export MATHMODEL_OS_SANDBOX_RUNTIME="$RUNTIME"
export MATHMODEL_OS_SANDBOX_IMAGE="$IMAGE"
export MATHMODEL_OS_SANDBOX_ATTESTATION_FILE="$OUT/attestation.json"
export MATHMODEL_OS_SANDBOX_ATTESTED=1
python3 -m pytest "$ROOT/tests/test_os_sandbox.py" "$ROOT/tests/test_solver_runtime.py" "$ROOT/tests/test_os_sandbox_integration.py" -q
