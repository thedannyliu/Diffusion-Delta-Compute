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
# Accept profiles via space-separated PROFILES or comma-separated PROFILES_CSV (preferred for sbatch)
PROFILES_CSV=${PROFILES_CSV:-"conservative,balanced,aggressive"}

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

# Capture an explicit exp_name, if provided on CLI, so we can suffix per-profile
BASE_EXP_NAME=""
for ((i=0; i<${#CLI_ARGS[@]}; i++)); do
  if [[ "${CLI_ARGS[$i]}" == "--exp_name" ]] && (( i+1 < ${#CLI_ARGS[@]} )); then
    BASE_EXP_NAME="${CLI_ARGS[$((i+1))]}"
    break
  fi
done

echo "[P2] Running rule-based gate with config: $CONFIG_PATH (python=$PYTHON)"
pushd "$PROJECT_ROOT" >/dev/null

# Build profile list (prefer PROFILES_CSV over PROFILES to avoid inherited env overriding)
if [[ -n "${PROFILES_CSV:-}" ]]; then
  IFS=',' read -r -a PROFILE_LIST <<<"$PROFILES_CSV"
elif [[ -n "${PROFILES:-}" ]]; then
  read -r -a PROFILE_LIST <<<"$PROFILES"
else
  IFS=',' read -r -a PROFILE_LIST <<<"conservative,balanced,aggressive"
fi

echo "[P2] Profiles to run: ${PROFILE_LIST[*]}"
set +e
for PROFILE_NAME in "${PROFILE_LIST[@]}"; do
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
  # Compose per-profile exp_name (preserve user-provided base if present)
  if [[ -n "$BASE_EXP_NAME" ]]; then
    EXP_NAME_COMBINED="${BASE_EXP_NAME}_${PROFILE_NAME}"
  else
    EXP_NAME_COMBINED="rule_gate_${PROFILE_NAME}"
  fi

  # Always pass --profile to downstream to aid reporting/config selection
  RUN_ARGS=(
    "${CLI_ARGS[@]}"
    "--profile" "$PROFILE_NAME"
    "--tau" "$TAU" "--rho" "$RHO" "--m" "$M_CONS" "--freeze_K" "$K_CONS"
    "--exp_name" "$EXP_NAME_COMBINED"
  )
  echo "[P2] PROFILE=$PROFILE_NAME tau=$TAU rho=$RHO m=$M_CONS K=$K_CONS"
  "$PYTHON" -m src.run "${RUN_ARGS[@]}"
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[P2] PROFILE=$PROFILE_NAME failed with exit code $rc; continuing..." >&2
  fi
done
set -e

popd >/dev/null
