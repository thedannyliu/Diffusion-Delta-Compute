#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

ENV_FEATURE_SOURCE=${FEATURE_SOURCE:-}
ENV_CONFIG_PATH=${CONFIG_PATH:-}

FEATURE_SOURCE=""
CONFIG_PATH=""

if [[ $# -gt 0 ]]; then
  FEATURE_SOURCE=$1
  shift
fi

if [[ $# -gt 0 ]]; then
  CONFIG_PATH=$1
  shift
fi

if [[ -z "$FEATURE_SOURCE" && -n "$ENV_FEATURE_SOURCE" ]]; then
  FEATURE_SOURCE="$ENV_FEATURE_SOURCE"
fi

if [[ -z "$CONFIG_PATH" && -n "$ENV_CONFIG_PATH" ]]; then
  CONFIG_PATH="$ENV_CONFIG_PATH"
fi

if [[ -z "$CONFIG_PATH" ]]; then
  CONFIG_PATH="$PROJECT_ROOT/configs/p3_smoke.yaml"
fi

ENGINE=${ENGINE:-}
OUT_DIR=${OUT_DIR:-}
WEIGHTS_OUT=${WEIGHTS_OUT:-$PROJECT_ROOT/models/gates/learned_gate_latest.npz}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-8}
TRAIN_BATCH=${TRAIN_BATCH:-1024}
TRAIN_LR=${TRAIN_LR:-1e-3}
TRAIN_POS_WEIGHT=${TRAIN_POS_WEIGHT:-1.0}

mkdir -p "$(dirname "$WEIGHTS_OUT")"

if [[ -z "$FEATURE_SOURCE" ]]; then
  mapfile -t FEATURE_CANDIDATES < <(find "$PROJECT_ROOT/reports" -path '*P1*teacher*features/teacher_features_*.npz' -print 2>/dev/null | sort)
  if [[ ${#FEATURE_CANDIDATES[@]} -eq 0 ]]; then
    echo "[P3] ERROR: No teacher feature files found. Provide a path or run P1 first." >&2
    exit 1
  fi
  FEATURE_SOURCE=${FEATURE_CANDIDATES[-1]}
fi

IFS=$'\n' read -r -a FEATURE_ARGS <<< "$FEATURE_SOURCE"
unset IFS

TRAIN_ARGS=(
  "--features" "${FEATURE_ARGS[@]}"
  "--out" "$WEIGHTS_OUT"
  "--epochs" "$TRAIN_EPOCHS"
  "--batch_size" "$TRAIN_BATCH"
  "--lr" "$TRAIN_LR"
  "--pos_weight" "$TRAIN_POS_WEIGHT"
)

if [[ -n ${TRAIN_THRESHOLD_GRID:-} ]]; then
  TRAIN_ARGS+=("--threshold_grid" "$TRAIN_THRESHOLD_GRID")
fi

echo "[P3] Training learned gate from feature file(s): $FEATURE_SOURCE"
pushd "$PROJECT_ROOT" >/dev/null
python -m src.learned.train_gate "${TRAIN_ARGS[@]}"
popd >/dev/null

echo "[P3] Running learned gate evaluation with weights: $WEIGHTS_OUT"
CLI_ARGS=("--mode" "learned_gate" "--config" "$CONFIG_PATH" "--learned_gate_weights" "$WEIGHTS_OUT")
if [[ -n "$ENGINE" ]]; then
  CLI_ARGS+=("--engine" "$ENGINE")
fi
if [[ -n "$OUT_DIR" ]]; then
  CLI_ARGS+=("--out_dir" "$OUT_DIR")
fi
CLI_ARGS+=("$@")

pushd "$PROJECT_ROOT" >/dev/null
python -m src.run "${CLI_ARGS[@]}"
popd >/dev/null
