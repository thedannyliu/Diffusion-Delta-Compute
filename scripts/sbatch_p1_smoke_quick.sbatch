#!/bin/bash
#SBATCH -J dcllm_p1_smoke_quick
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=40G
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
export WANDB_MODE=${WANDB_MODE:-disabled}
mkdir -p logs

CONFIG_PATH=${CONFIG_PATH:-configs/p1_smoke.yaml}
ENGINE=${ENGINE:-d2f}
OUT_DIR=${OUT_DIR:-reports_quick}

declare -a DATASETS=(
  "wikitext:wikitext_smoke_quick"
  "lambada:lambada_smoke_quick"
  "gsm8k:gsm8k_smoke_quick"
)

for spec in "${DATASETS[@]}"; do
  IFS=':' read -r TASK EPNAME <<<"$spec"
  echo "[sbatch][P1-quick] task=${TASK}"
  srun --exclusive -N1 -n1 \
    bash scripts/run_p1_teacher.sh "$CONFIG_PATH" \
      --engine "$ENGINE" \
      --task "$TASK" \
      --num_prompts 3 \
      --out_dir "$OUT_DIR" \
      --exp_name "${EPNAME}" \
      --oracle_eval \
      --lte_probe heun

done
