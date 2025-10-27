#!/bin/bash
#SBATCH -J dcllm_p1_smoke_all
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH -t 3:00:00
#SBATCH -o logs/%x-%j.out

# Prefer PyTorch wheel-bundled CUDA libs; keep module load but let our LD_LIBRARY_PATH take precedence later
module load cuda/12.1 || true

# Unset PYTHONPATH to avoid conflicts
unset PYTHONPATH

# Activate conda
DEFAULT_CONDA_ROOT="/usr/local/pace-apps/manual/packages/anaconda3/2023.03"
DEFAULT_CONDA_ENV="/storage/ice1/2/9/eliu354/conda_envs/dcllm"
declare -a CONDA_CANDIDATES=()
if [[ -n "${CONDA_ROOT:-}" ]]; then
  CONDA_CANDIDATES+=("$CONDA_ROOT")
fi
CONDA_CANDIDATES+=("$DEFAULT_CONDA_ROOT" "$HOME/miniconda3")
for candidate in "${CONDA_CANDIDATES[@]}"; do
  if [[ -f "$candidate/etc/profile.d/conda.sh" ]] && [[ -x "$candidate/bin/conda" ]] && "$candidate/bin/conda" --version >/dev/null 2>&1; then
    CONDA_ROOT="$candidate"
    # shellcheck disable=SC1090
    source "$candidate/etc/profile.d/conda.sh"
    break
  fi
done
if [[ -z "${CONDA_ROOT:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    echo "[ERROR] conda initialization script not found; export CONDA_ROOT or install Miniconda." >&2
    exit 1
  fi
fi

CONDA_ENV_TARGET=${CONDA_ENV_TARGET:-}
if [[ -z "$CONDA_ENV_TARGET" ]]; then
  if [[ -d "$DEFAULT_CONDA_ENV" ]]; then
    CONDA_ENV_TARGET="$DEFAULT_CONDA_ENV"
  else
    CONDA_ENV_TARGET="dcllm"
  fi
fi

if ! conda activate "$CONDA_ENV_TARGET" >/dev/null 2>&1; then
  echo "[ERROR] failed to activate conda environment '$CONDA_ENV_TARGET'." >&2
  exit 1
fi

export WANDB_MODE=${WANDB_MODE:-online}
# Skip torchvision entirely to avoid import-time issues in transformers
export TRANSFORMERS_NO_TORCHVISION=1
# Ensure PyTorch wheel CUDA libs are first in search path to avoid symbol mismatches (e.g., __nvJitLinkComplete_12_4)
if [[ -n "$CONDA_PREFIX" ]]; then
  NV_ROOT="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$NV_ROOT/nvjitlink/lib:$NV_ROOT/cublas/lib:$NV_ROOT/cusparse/lib:$NV_ROOT/cudnn/lib:$NV_ROOT/cusolver/lib:$NV_ROOT/cufft/lib:$NV_ROOT/curand/lib:$NV_ROOT/nvtx/lib:${LD_LIBRARY_PATH:-}"
fi
mkdir -p logs

export PYTHON="/storage/ice1/2/9/eliu354/conda_envs/dcllm/bin/python"

CONFIG_PATH=${CONFIG_PATH:-configs/p1_smoke.yaml}
ENGINE=${ENGINE:-d2f}
DATA_ROOT=${DATA_ROOT:-}

declare -a DATASETS=(
  "gsm8k:10:gsm8k_smoke"
)

echo "[sbatch] P1 smoke triplet starting (engine=${ENGINE}) from $(realpath "$0")"
for spec in "${DATASETS[@]}"; do
  IFS=':' read -r TASK N EPNAME <<<"$spec"
echo "[sbatch][P1] Full run task=${TASK} n=${N} (CONDA_ROOT=${CONDA_ROOT:-unset} CONDA_PREFIX=${CONDA_PREFIX:-unset} PYTHON=$(command -v python))"
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
