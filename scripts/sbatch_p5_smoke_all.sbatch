#!/bin/bash
#SBATCH -J dcllm_p5_smoke_all
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 06:00:00
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

CONFIG_ORACLE=${CONFIG_ORACLE:-configs/p5_smoke.yaml}
CONFIG_BASE=${CONFIG_BASE:-configs/p5_smoke.yaml}
ENGINE=${ENGINE:-d2f}

declare -a DATASETS=(
  "wikitext:100:wikitext_p5"
  "lambada:100:lambada_p5"
  "gsm8k:100:gsm8k_p5"
)

for spec in "${DATASETS[@]}"; do
  IFS=':' read -r TASK N EPNAME <<<"$spec"
  echo "[sbatch][P5] Oracle task=${TASK} n=${N}"
  srun --exclusive -N1 -n1 \
    bash scripts/run_p5_oracle.sh "$CONFIG_ORACLE" \
      --engine "$ENGINE" \
      --task "$TASK" \
      --num_prompts "$N" \
      --exp_name "${EPNAME}_oracle"

  echo "[sbatch][P5] Baselines task=${TASK} n=${N}"
  srun --exclusive -N1 -n1 \
    bash scripts/run_p5_baselines.sh "$CONFIG_BASE" \
      --engine "$ENGINE" \
      --task "$TASK" \
      --num_prompts "$N" \
      --exp_name "${EPNAME}_baselines"
done

echo "[sbatch] P5 smoke triplet completed"
