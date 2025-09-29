#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
REPORT_ROOT=${REPORT_ROOT:-$PROJECT_ROOT/reports}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=${OUT_DIR:-$REPORT_ROOT/P6/combine/$TIMESTAMP}
RUNS_DIR="$OUT_DIR/runs"
TABLES_DIR="$OUT_DIR/tables"

mkdir -p "$RUNS_DIR" "$TABLES_DIR"

export REPORT_ROOT OUT_DIR RUNS_DIR TABLES_DIR

python - <<'PY'
import json
import csv
from pathlib import Path
from datetime import datetime
import os

report_root = Path(os.environ['REPORT_ROOT']).resolve()
out_dir = Path(os.environ['OUT_DIR']).resolve()
runs_dir = Path(os.environ['RUNS_DIR']).resolve()
tables_dir = Path(os.environ['TABLES_DIR']).resolve()

patterns = [
    ("teacher", "**/teacher_summary_*.json"),
    ("rule_gate", "**/rule_gate_summary_*.json"),
    ("learned_gate", "**/learned_gate_summary_*.json"),
    ("adaptive", "**/adaptive_summary_*.json"),
    ("oracle", "**/oracle_summary_*.json"),
    ("baselines", "**/baselines_*.json"),
]

summaries = {}
for name, pattern in patterns:
    candidates = sorted(report_root.rglob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        continue
    latest = candidates[-1]
    try:
        data = json.loads(latest.read_text())
    except Exception:
        continue
    summaries[name] = {
        "path": str(latest.relative_to(report_root)),
        "updated_at": datetime.fromtimestamp(latest.stat().st_mtime).isoformat(),
        "payload": data,
    }

combined = {
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "report_root": str(report_root),
    "summaries": summaries,
}

# 寫入綜合 JSON
(out_dir / "runs" / "final_metrics.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False))

# 產出簡易表格
rows = []
for name, entry in summaries.items():
    payload = entry["payload"]
    rows.append({
        "mode": name,
        "latency_ms_mean": payload.get("latency_ms_mean", ""),
        "skip_ratio_mean": payload.get("skip_ratio_mean", payload.get("skip_ratio_est_mean", "")),
        "risk_delta": payload.get("risk_delta", ""),
        "delta_violation_mean": payload.get("delta_violation_mean", ""),
        "source_path": entry["path"],
    })

if rows:
    fieldnames = ["mode", "latency_ms_mean", "skip_ratio_mean", "risk_delta", "delta_violation_mean", "source_path"]
    with open(tables_dir / "summary_overview.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

print(f"[P6] Combined summary written to {out_dir}")
PY
