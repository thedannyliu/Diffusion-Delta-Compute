#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
if [[ $# -gt 0 ]]; then
  CONFIG_PATH=$1
  shift
else
  CONFIG_PATH="$PROJECT_ROOT/configs/p5_smoke.yaml"
fi

ENGINE=${ENGINE:-}
OUT_DIR=${OUT_DIR:-}
TASKS_CFG=${TASKS_CFG:-}
PATHS_CFG=${PATHS_CFG:-}
DATA_ROOT=${DATA_ROOT:-}
MAX_NEW=${MAX_NEW:-}

CLI_ARGS=("--mode" "oracle" "--config" "$CONFIG_PATH")
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
if [[ -n "$DATA_ROOT" ]]; then
  CLI_ARGS+=("--data_root" "$DATA_ROOT")
fi
if [[ -n "$MAX_NEW" ]]; then
  CLI_ARGS+=("--max_new_tokens" "$MAX_NEW")
fi

CLI_ARGS+=("$@")

echo "[P5] Running oracle skip/freeze analysis with config: $CONFIG_PATH (python=$PYTHON)"
pushd "$PROJECT_ROOT" >/dev/null
"$PYTHON" -m src.run "${CLI_ARGS[@]}"
popd >/dev/null
