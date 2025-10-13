#!/bin/bash
#SBATCH -J dcllm_p2_smoke_quick
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=60G
#SBATCH -t 01:30:00
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
export TRANSFORMERS_NO_TORCHVISION=1
if [[ -n "$CONDA_PREFIX" ]]; then
  NV_ROOT="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
  export LD_LIBRARY_PATH="$NV_ROOT/nvjitlink/lib:$NV_ROOT/cublas/lib:$NV_ROOT/cusparse/lib:$NV_ROOT/cudnn/lib:$NV_ROOT/cusolver/lib:$NV_ROOT/cufft/lib:$NV_ROOT/curand/lib:$NV_ROOT/nvtx/lib:${LD_LIBRARY_PATH:-}"
fi
mkdir -p logs

CONFIG_PATH=${CONFIG_PATH:-configs/p2_smoke.yaml}
ENGINE=${ENGINE:-d2f}
OUT_DIR=${OUT_DIR:-reports_quick}
DATA_ROOT=${DATA_ROOT:-}

# spec: TASK:PROFILE:RISK_DELTA:EXP_NAME
declare -a DATASETS=(
  "gsm8k:balanced:0.010:gsm8k_rule_quick"
)

echo "[sbatch] P2 quick smoke starting (engine=${ENGINE})"
for spec in "${DATASETS[@]}"; do
  IFS=':' read -r TASK PROFILE RDELTA EPNAME <<<"$spec"
  echo "[sbatch][P2-quick] task=${TASK} profile=${PROFILE} δ=${RDELTA}"
  srun --exclusive -N1 -n1 \
    env OUT_DIR="$OUT_DIR" PROFILES_CSV="$PROFILE" ${DATA_ROOT:+DATA_ROOT="$DATA_ROOT"} \
    bash scripts/run_p2_rule_gate.sh "$CONFIG_PATH" \
      --engine "$ENGINE" \
      --task "$TASK" \
      --num_prompts 5 \
      ${DATA_ROOT:+--data_root "$DATA_ROOT"} \
      --max_new_tokens 128 \
      --risk_delta "$RDELTA" \
      --budget_fraction 0.60 \
      --exp_name "$EPNAME"
done

echo "[sbatch] P2 quick smoke completed"
