#!/bin/bash
#SBATCH -J dcllm_p4_smoke_all
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 12:00:00
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

CONFIG_PATH=${CONFIG_PATH:-configs/p4_smoke.yaml}
ENGINE=${ENGINE:-d2f}

declare -a DATASETS=(
  "wikitext:2000:wikitext_adapt"
  "lambada:2000:lambada_adapt"
  "gsm8k:1500:gsm8k_adapt"
)

for spec in "${DATASETS[@]}"; do
  IFS=':' read -r TASK N EPNAME <<<"$spec"
  echo "[sbatch][P4] Full run task=${TASK} n=${N}"
  srun --exclusive -N1 -n1 \
    bash scripts/run_p4_adaptive.sh "$CONFIG_PATH" \
      --engine "$ENGINE" \
      --task "$TASK" \
      --num_prompts "$N" \
      --exp_name "$EPNAME" \
      --lte_probe heun \
      --lte_eps 0.015 --lte_min_consec 2 --max_stride 4 --adaptive_budget 0.20

  echo "[sbatch][P4] Trace5 task=${TASK}"
  srun --exclusive -N1 -n1 \
    bash scripts/run_p4_adaptive.sh "$CONFIG_PATH" \
      --engine "$ENGINE" \
      --task "$TASK" \
      --num_prompts 5 \
      --exp_name "${EPNAME}_trace5" \
      --lte_probe heun \
      --lte_eps 0.015 --lte_min_consec 2 --max_stride 4 --adaptive_budget 0.20
done

echo "[sbatch] P4 smoke triplet completed"

