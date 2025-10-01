#!/bin/bash
#SBATCH -J dcllm_p2_rule
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

CONFIG_PATH=${CONFIG_PATH:-configs/p2_smoke.yaml}
ENGINE=${ENGINE:-d2f}
OUT_DIR=${OUT_DIR:-}
TASKS_CFG=${TASKS_CFG:-}
PATHS_CFG=${PATHS_CFG:-}
THRESHOLDS_CFG=${THRESHOLDS_CFG:-}
PROFILE=${PROFILE:-}
DATASET_KEY=${DATASET_KEY:-}
RISK_DELTA=${RISK_DELTA:-}
BUDGET_FRACTION=${BUDGET_FRACTION:-}
LAYER_M=${LAYER_M:-}

echo "[sbatch] P2 rule gate sweep using config=${CONFIG_PATH} engine=${ENGINE}"
PROFILES=${PROFILES:-"conservative balanced aggressive"}
EXPORT_VARS=ALL,ENGINE=${ENGINE},OUT_DIR=${OUT_DIR},TASKS_CFG=${TASKS_CFG},PATHS_CFG=${PATHS_CFG},THRESHOLDS_CFG=${THRESHOLDS_CFG},PROFILE=${PROFILE},DATASET_KEY=${DATASET_KEY},RISK_DELTA=${RISK_DELTA},BUDGET_FRACTION=${BUDGET_FRACTION},LAYER_M=${LAYER_M},PROFILES="${PROFILES}"
srun --export=${EXPORT_VARS} bash scripts/run_p2_rule_gate.sh "$CONFIG_PATH" "$@"
