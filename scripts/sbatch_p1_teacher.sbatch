#!/bin/bash
#SBATCH -J dcllm_p1_teacher
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h100:1
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

CONFIG_PATH=${CONFIG_PATH:-configs/p1_smoke.yaml}
ENGINE=${ENGINE:-d2f}
OUT_DIR=${OUT_DIR:-}

echo "[sbatch] P1 teacher run using config=${CONFIG_PATH} engine=${ENGINE}"
SRUN_PY=$(command -v python || true)
if [[ -n "$SRUN_PY" ]]; then
  srun --export=ALL,ENGINE=${ENGINE},OUT_DIR=${OUT_DIR},PYTHON="$SRUN_PY" bash scripts/run_p1_teacher.sh "$CONFIG_PATH" "$@"
else
  srun --export=ALL,ENGINE=${ENGINE},OUT_DIR=${OUT_DIR} bash scripts/run_p1_teacher.sh "$CONFIG_PATH" "$@"
fi
