#!/bin/bash
#SBATCH -J dcllm_p1_smoke_all
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH -t 3:00:00
#SBATCH -o logs/%x-%j.out

# Prefer PyTorch wheel-bundled CUDA libs; keep module load but let our LD_LIBRARY_PATH take precedence later
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
# Skip torchvision entirely to avoid import-time issues in transformers
export TRANSFORMERS_NO_TORCHVISION=1
# Ensure PyTorch wheel CUDA libs are first in search path to avoid symbol mismatches (e.g., __nvJitLinkComplete_12_4)
if [[ -n "$CONDA_PREFIX" ]]; then
  NV_ROOT="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$NV_ROOT/nvjitlink/lib:$NV_ROOT/cublas/lib:$NV_ROOT/cusparse/lib:$NV_ROOT/cudnn/lib:$NV_ROOT/cusolver/lib:$NV_ROOT/cufft/lib:$NV_ROOT/curand/lib:$NV_ROOT/nvtx/lib:${LD_LIBRARY_PATH:-}"
fi
mkdir -p logs

CONFIG_PATH=${CONFIG_PATH:-configs/p1_smoke.yaml}
ENGINE=${ENGINE:-d2f}
DATA_ROOT=${DATA_ROOT:-}

declare -a DATASETS=(
  "gsm8k:50:gsm8k_smoke"
)

echo "[sbatch] P1 smoke triplet starting (engine=${ENGINE})"
for spec in "${DATASETS[@]}"; do
  IFS=':' read -r TASK N EPNAME <<<"$spec"
  echo "[sbatch][P1] Full run task=${TASK} n=${N}"
  srun --exclusive -N1 -n1 \
    bash scripts/run_p1_teacher.sh "$CONFIG_PATH" \
      --engine "$ENGINE" \
      --task "$TASK" \
      --num_prompts "$N" \
      ${DATA_ROOT:+--data_root "$DATA_ROOT"} \
      --max_new_tokens 256 \
      --exp_name "$EPNAME" \
      --oracle_eval \
      --lte_probe heun \
      --dump_features_dir reports/artifacts/features

  # echo "[sbatch][P1] Trace5 task=${TASK}"
  # srun --exclusive -N1 -n1 \
  #   bash scripts/run_p1_teacher.sh "$CONFIG_PATH" \
  #     --engine "$ENGINE" \
  #     --task "$TASK" \
  #     --num_prompts "$N" \
  #     --exp_name "${EPNAME}_trace5" \
  #     --consistency_full_compute \
  #     --lte_probe heun
done

echo "[sbatch] P1 smoke triplet completed"
