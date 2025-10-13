#!/bin/bash
#SBATCH -J dcllm_p4_conservative
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 02:30:00
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
export TRANSFORMERS_NO_TORCHVISION=1
if [[ -n "$CONDA_PREFIX" ]]; then
  NV_ROOT="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
  export LD_LIBRARY_PATH="$NV_ROOT/nvjitlink/lib:$NV_ROOT/cublas/lib:$NV_ROOT/cusparse/lib:$NV_ROOT/cudnn/lib:$NV_ROOT/cusolver/lib:$NV_ROOT/cufft/lib:$NV_ROOT/curand/lib:$NV_ROOT/nvtx/lib:${LD_LIBRARY_PATH:-}"
fi
mkdir -p logs

CONFIG_PATH=${CONFIG_PATH:-configs/p4_smoke.yaml}
ENGINE=${ENGINE:-d2f}

# 可覆寫的資料根目錄（若提供，將強制從本地載入固定順序的資料）
DATA_ROOT=${DATA_ROOT:-}

declare -a DATASETS=(
  "gsm8k:200:gsm8k_adapt_conservative"
)

for spec in "${DATASETS[@]}"; do
  IFS=':' read -r TASK N EPNAME <<<"$spec"
  echo "[sbatch][P4-conservative] task=${TASK} n=${N}"
  srun --exclusive -N1 -n1 \
    env ${DATA_ROOT:+DATA_ROOT="$DATA_ROOT"} MAX_NEW=256 HARD_BUDGET=1 \
    bash scripts/run_p4_adaptive.sh "$CONFIG_PATH" \
      --engine "$ENGINE" \
      --task "$TASK" \
      --num_prompts "$N" \
      --exp_name "$EPNAME" \
      --lte_probe heun \
      --lte_eps 0.010 --lte_min_consec 3 --max_stride 3 \
      --adaptive_budget 0.20 \
      --risk_delta 0.02
done

echo "[sbatch] P4 conservative job completed"
