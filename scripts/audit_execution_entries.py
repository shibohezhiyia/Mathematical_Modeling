from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.execution_entry_audit import audit_execution_entries

parser = argparse.ArgumentParser(description="Audit numerical execution entry points")
parser.add_argument("root", nargs="?", type=Path, default=ROOT)
args = parser.parse_args()
print(json.dumps(audit_execution_entries(args.root), ensure_ascii=False, indent=2))
