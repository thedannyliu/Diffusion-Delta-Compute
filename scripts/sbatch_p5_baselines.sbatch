#!/bin/bash
#SBATCH -J dcllm_p5_base
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
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
export WANDB_MODE=${WANDB_MODE:-offline}
mkdir -p logs

CONFIG_PATH=${CONFIG_PATH:-configs/p5_smoke.yaml}
OUT_DIR=${OUT_DIR:-}
TASKS_CFG=${TASKS_CFG:-}
PATHS_CFG=${PATHS_CFG:-}
NUM_STEPS=${NUM_STEPS:-}

echo "[sbatch] P5 baselines run using config=${CONFIG_PATH}"
EXPORT_VARS=ALL,OUT_DIR=${OUT_DIR},TASKS_CFG=${TASKS_CFG},PATHS_CFG=${PATHS_CFG},NUM_STEPS=${NUM_STEPS}
srun --export=${EXPORT_VARS} bash scripts/run_p5_baselines.sh "$CONFIG_PATH" "$@"
