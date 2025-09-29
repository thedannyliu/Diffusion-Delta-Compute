#!/bin/bash
#SBATCH -J dcllm_p3_learned
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out

module load cuda/12.1 || true

CONDA_ROOT=${CONDA_ROOT:-$HOME/miniconda3}
if [[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "[ERROR] conda initialization script not found; set CONDA_ROOT or install Miniconda." >&2
  exit 1
fi

if ! conda activate dcllm >/dev/null 2>&1; then
  echo "[ERROR] failed to activate conda environment 'dcllm'." >&2
  exit 1
fi

export PYTHON="${PYTHON:-$(command -v python)}"
export WANDB_MODE=${WANDB_MODE:-online}
mkdir -p logs

FEATURE_SOURCE=${FEATURE_SOURCE:-}
CONFIG_PATH=${CONFIG_PATH:-configs/p3_smoke.yaml}
ENGINE=${ENGINE:-d2f}
OUT_DIR=${OUT_DIR:-}
TASKS_CFG=${TASKS_CFG:-}
PATHS_CFG=${PATHS_CFG:-}
THRESHOLDS_CFG=${THRESHOLDS_CFG:-}
PROFILE=${PROFILE:-}
DATASET_KEY=${DATASET_KEY:-}
RISK_DELTA=${RISK_DELTA:-}
BUDGET_FRACTION=${BUDGET_FRACTION:-}
WEIGHTS_OUT=${WEIGHTS_OUT:-models/gates/learned_gate_latest.npz}

if [[ -n "$FEATURE_SOURCE" ]]; then
  echo "[sbatch] Using feature glob: $FEATURE_SOURCE"
else
  echo "[sbatch] Using latest P1 features (auto-discovery)"
fi

RUN_ARGS=()
if [[ -n "$FEATURE_SOURCE" ]]; then
  RUN_ARGS+=("$FEATURE_SOURCE")
fi

EXPORT_VARS=ALL,FEATURE_SOURCE=${FEATURE_SOURCE},CONFIG_PATH=${CONFIG_PATH},ENGINE=${ENGINE},OUT_DIR=${OUT_DIR},WEIGHTS_OUT=${WEIGHTS_OUT},TASKS_CFG=${TASKS_CFG},PATHS_CFG=${PATHS_CFG},THRESHOLDS_CFG=${THRESHOLDS_CFG},PROFILE=${PROFILE},DATASET_KEY=${DATASET_KEY},RISK_DELTA=${RISK_DELTA},BUDGET_FRACTION=${BUDGET_FRACTION}
srun --export=${EXPORT_VARS} bash scripts/run_p3_learned_gate.sh "${RUN_ARGS[@]}" "$@"
