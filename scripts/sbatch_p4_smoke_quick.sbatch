#!/bin/bash
#SBATCH -J dcllm_p4_smoke_quick
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=60G
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out

module load cuda/12.1 || true

# Activate conda
source /storage/ice1/2/9/eliu354/miniconda3/etc/profile.d/conda.sh
conda activate dcllm

export WANDB_MODE=${WANDB_MODE:-disabled}
export TRANSFORMERS_NO_TORCHVISION=1
if [[ -n "$CONDA_PREFIX" ]]; then
  NV_ROOT="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
  export LD_LIBRARY_PATH="$NV_ROOT/nvjitlink/lib:$NV_ROOT/cublas/lib:$NV_ROOT/cusparse/lib:$NV_ROOT/cudnn/lib:$NV_ROOT/cusolver/lib:$NV_ROOT/cufft/lib:$NV_ROOT/curand/lib:$NV_ROOT/nvtx/lib:${LD_LIBRARY_PATH:-}"
fi
mkdir -p logs

CONFIG_PATH=${CONFIG_PATH:-configs/p4_smoke.yaml}
ENGINE=${ENGINE:-d2f}
OUT_DIR=${OUT_DIR:-reports_quick}
DATA_ROOT=${DATA_ROOT:-}
MAX_NEW=${MAX_NEW:-128}

declare -a DATASETS=(
  "gsm8k:5:gsm8k_adapt_quick"
)

for spec in "${DATASETS[@]}"; do
  IFS=':' read -r TASK N EPNAME <<<"$spec"
  echo "[sbatch][P4-quick] task=${TASK} n=${N}"
  srun --exclusive -N1 -n1 \
    env OUT_DIR="$OUT_DIR" ${DATA_ROOT:+DATA_ROOT="$DATA_ROOT"} MAX_NEW="$MAX_NEW" \
    bash scripts/run_p4_adaptive.sh "$CONFIG_PATH" \
      --engine "$ENGINE" \
      --task "$TASK" \
      --num_prompts "$N" \
      --exp_name "$EPNAME" \
      --lte_probe heun \
      --lte_eps 0.015 --lte_min_consec 2 --max_stride 4 --adaptive_budget 0.20
done

echo "[sbatch] P4 quick smoke completed"
