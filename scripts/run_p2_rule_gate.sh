#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
if [[ $# -gt 0 ]]; then
  CONFIG_PATH=$1
  shift
else
  CONFIG_PATH="$PROJECT_ROOT/configs/p2_smoke.yaml"
fi

ENGINE=${ENGINE:-}
OUT_DIR=${OUT_DIR:-}
TASKS_CFG=${TASKS_CFG:-}
PATHS_CFG=${PATHS_CFG:-}
THRESHOLDS_CFG=${THRESHOLDS_CFG:-}
PROFILE=${PROFILE:-}
DATASET_KEY=${DATASET_KEY:-}
RISK_DELTA=${RISK_DELTA:-}
BUDGET_FRACTION=${BUDGET_FRACTION:-}
LAYER_M=${LAYER_M:-}
MODE=rule_gate
PROFILES=${PROFILES:-"conservative balanced aggressive"}

CLI_ARGS=("--mode" "$MODE" "--config" "$CONFIG_PATH")
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
if [[ -n "$THRESHOLDS_CFG" ]]; then
  CLI_ARGS+=("--thresholds_cfg" "$THRESHOLDS_CFG")
fi
if [[ -n "$PROFILE" ]]; then
  CLI_ARGS+=("--profile" "$PROFILE")
fi
if [[ -n "$DATASET_KEY" ]]; then
  CLI_ARGS+=("--dataset" "$DATASET_KEY")
fi
if [[ -n "$RISK_DELTA" ]]; then
  CLI_ARGS+=("--risk_delta" "$RISK_DELTA")
fi
if [[ -n "$BUDGET_FRACTION" ]]; then
  CLI_ARGS+=("--budget_fraction" "$BUDGET_FRACTION")
fi
if [[ -n "$LAYER_M" ]]; then
  CLI_ARGS+=("--layer_recompute_M" "$LAYER_M")
fi

CLI_ARGS+=("$@")

echo "[P2] Running rule-based gate with config: $CONFIG_PATH (python=$PYTHON)"
pushd "$PROJECT_ROOT" >/dev/null

for PROFILE_NAME in $PROFILES; do
  case "$PROFILE_NAME" in
    conservative)
      TAU=${TAU_CONSERVATIVE:-0.97}
      RHO=${RHO_CONSERVATIVE:-0.05}
      M_CONS=${M_CONSERVATIVE:-2}
      K_CONS=${K_CONSERVATIVE:-1}
      ;;
    balanced)
      TAU=${TAU_BALANCED:-0.95}
      RHO=${RHO_BALANCED:-0.10}
      M_CONS=${M_BALANCED:-2}
      K_CONS=${K_BALANCED:-2}
      ;;
    aggressive)
      TAU=${TAU_AGGRESSIVE:-0.90}
      RHO=${RHO_AGGRESSIVE:-0.20}
      M_CONS=${M_AGGRESSIVE:-1}
      K_CONS=${K_AGGRESSIVE:-3}
      ;;
    *)
      echo "[WARN] Unknown PROFILE '$PROFILE_NAME', skipping." >&2
      continue
      ;;
  esac
  RUN_ARGS=("${CLI_ARGS[@]}" "--tau" "$TAU" "--rho" "$RHO" "--m" "$M_CONS" "--freeze_K" "$K_CONS" "--exp_name" "wikitext_rule_gate_$PROFILE_NAME")
  echo "[P2] PROFILE=$PROFILE_NAME tau=$TAU rho=$RHO m=$M_CONS K=$K_CONS"
  "$PYTHON" -m src.run "${RUN_ARGS[@]}"
done

popd >/dev/null
