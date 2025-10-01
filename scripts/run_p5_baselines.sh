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

OUT_DIR=${OUT_DIR:-}
TASKS_CFG=${TASKS_CFG:-}
PATHS_CFG=${PATHS_CFG:-}
NUM_STEPS=${NUM_STEPS:-}

CLI_ARGS=("--mode" "baselines" "--config" "$CONFIG_PATH")
if [[ -n "$OUT_DIR" ]]; then
  CLI_ARGS+=("--out_dir" "$OUT_DIR")
fi
if [[ -n "$TASKS_CFG" ]]; then
  CLI_ARGS+=("--tasks_cfg" "$TASKS_CFG")
fi
if [[ -n "$PATHS_CFG" ]]; then
  CLI_ARGS+=("--paths_cfg" "$PATHS_CFG")
fi
if [[ -n "$NUM_STEPS" ]]; then
  CLI_ARGS+=("--num_steps" "$NUM_STEPS")
fi

CLI_ARGS+=("$@")

echo "[P5] Generating baseline policies with config: $CONFIG_PATH (python=$PYTHON)"
pushd "$PROJECT_ROOT" >/dev/null
"$PYTHON" -m src.run "${CLI_ARGS[@]}"
popd >/dev/null
