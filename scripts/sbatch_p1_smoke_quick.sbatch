#!/bin/bash
#SBATCH -J dcllm_p1_smoke_quick
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out

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

export PYTHON="${PYTHON:-$(command -v python)}"
export WANDB_MODE=${WANDB_MODE:-disabled}
export TRANSFORMERS_NO_TORCHVISION=1
if [[ -n "$CONDA_PREFIX" ]]; then
  NV_ROOT="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$NV_ROOT/nvjitlink/lib:$NV_ROOT/cublas/lib:$NV_ROOT/cusparse/lib:$NV_ROOT/cudnn/lib:$NV_ROOT/cusolver/lib:$NV_ROOT/cufft/lib:$NV_ROOT/curand/lib:$NV_ROOT/nvtx/lib:${LD_LIBRARY_PATH:-}"
fi
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
      --lte_probe heun

done
