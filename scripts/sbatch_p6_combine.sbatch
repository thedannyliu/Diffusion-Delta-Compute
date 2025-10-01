#!/bin/bash
#SBATCH -J dcllm_p6_combine
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out

module load cuda/12.1 || true

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

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
mkdir -p logs

REPORT_ROOT=${REPORT_ROOT:-$PROJECT_ROOT/reports}
OUT_DIR=${OUT_DIR:-}

if [[ -z "$OUT_DIR" ]]; then
  export OUT_DIR="$REPORT_ROOT/P6/combine/$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUT_DIR"

echo "[sbatch] P6 combine report using REPORT_ROOT=${REPORT_ROOT} -> OUT_DIR=${OUT_DIR}"
EXPORT_VARS=ALL,REPORT_ROOT=${REPORT_ROOT},OUT_DIR=${OUT_DIR}
srun --export=${EXPORT_VARS} bash scripts/run_p6_combine.sh "$@"
