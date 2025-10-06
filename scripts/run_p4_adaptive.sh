#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
if [[ $# -gt 0 ]]; then
  CONFIG_PATH=$1
  shift
else
  CONFIG_PATH="$PROJECT_ROOT/configs/p4_smoke.yaml"
fi

ENGINE=${ENGINE:-}
OUT_DIR=${OUT_DIR:-}
TASKS_CFG=${TASKS_CFG:-}
PATHS_CFG=${PATHS_CFG:-}
RISK_DELTA=${RISK_DELTA:-}
LTE_EPS=${LTE_EPS:-}
LTE_MIN_CONSEC=${LTE_MIN_CONSEC:-}
MAX_STRIDE=${MAX_STRIDE:-}
ADAPTIVE_BUDGET=${ADAPTIVE_BUDGET:-}
DATA_ROOT=${DATA_ROOT:-}
MAX_NEW=${MAX_NEW:-}
HARD_BUDGET=${HARD_BUDGET:-}

CLI_ARGS=("--mode" "adaptive" "--config" "$CONFIG_PATH")
if [[ -n "$ENGINE" ]]; then
  CLI_ARGS+=("--engine" "$ENGINE")
fi
if [[ -n "$OUT_DIR" ]]; then
  CLI_ARGS+=("--out_dir" "$OUT_DIR")
fi
if [[ -n "$TASKS_CFG" ]]; then
  CLI_ARGS+=("--tasks_cfg" "$TASKS_CFG")
fi
if [[ -n "$PATHS_CFG" ]]; then
  CLI_ARGS+=("--paths_cfg" "$PATHS_CFG")
fi
if [[ -n "$RISK_DELTA" ]]; then
  CLI_ARGS+=("--risk_delta" "$RISK_DELTA")
fi
if [[ -n "$LTE_EPS" ]]; then
  CLI_ARGS+=("--lte_eps" "$LTE_EPS")
fi
if [[ -n "$LTE_MIN_CONSEC" ]]; then
  CLI_ARGS+=("--lte_min_consec" "$LTE_MIN_CONSEC")
fi
if [[ -n "$MAX_STRIDE" ]]; then
  CLI_ARGS+=("--max_stride" "$MAX_STRIDE")
fi
if [[ -n "$ADAPTIVE_BUDGET" ]]; then
  CLI_ARGS+=("--adaptive_budget" "$ADAPTIVE_BUDGET")
fi
if [[ -n "$DATA_ROOT" ]]; then
  CLI_ARGS+=("--data_root" "$DATA_ROOT")
fi
if [[ -n "$MAX_NEW" ]]; then
  CLI_ARGS+=("--max_new_tokens" "$MAX_NEW")
fi
if [[ -n "$HARD_BUDGET" ]]; then
  CLI_ARGS+=("--adaptive_hard_budget")
fi

CLI_ARGS+=("$@")

echo "[P4] Running adaptive stride scheduling with config: $CONFIG_PATH (python=$PYTHON)"
pushd "$PROJECT_ROOT" >/dev/null
"$PYTHON" -m src.run "${CLI_ARGS[@]}"
popd >/dev/null
