#!/bin/bash
#SBATCH -J dcllm_p2_smoke_all
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
export TRANSFORMERS_NO_TORCHVISION=1
if [[ -n "$CONDA_PREFIX" ]]; then
  NV_ROOT="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
  export LD_LIBRARY_PATH="$NV_ROOT/nvjitlink/lib:$NV_ROOT/cublas/lib:$NV_ROOT/cusparse/lib:$NV_ROOT/cudnn/lib:$NV_ROOT/cusolver/lib:$NV_ROOT/cufft/lib:$NV_ROOT/curand/lib:$NV_ROOT/nvtx/lib:${LD_LIBRARY_PATH:-}"
fi
mkdir -p logs

CONFIG_PATH=${CONFIG_PATH:-configs/p2_smoke.yaml}
ENGINE=${ENGINE:-d2f}

# spec: TASK:N_PROMPTS:PROFILE:RISK_DELTA:EXP
declare -a DATASETS=(
  "gsm8k:200:balanced:0.010:gsm8k_rule"
)

echo "[sbatch] P2 smoke triplet starting (engine=${ENGINE})"
for spec in "${DATASETS[@]}"; do
  IFS=':' read -r TASK N PROFILE RDELTA EPNAME <<<"$spec"
  echo "[sbatch][P2] Full run task=${TASK} n=${N} profile=${PROFILE} δ=${RDELTA}"
  srun --exclusive -N1 -n1 \
    bash scripts/run_p2_rule_gate.sh "$CONFIG_PATH" \
      --engine "$ENGINE" \
      --task "$TASK" \
      --num_prompts "$N" \
      --profile "$PROFILE" \
      --risk_delta "$RDELTA" \
      --budget_fraction 0.60 \
      --exp_name "$EPNAME"

  echo "[sbatch][P2] Trace5 task=${TASK}"
  srun --exclusive -N1 -n1 \
    bash scripts/run_p2_rule_gate.sh "$CONFIG_PATH" \
      --engine "$ENGINE" \
      --task "$TASK" \
      --num_prompts 5 \
      --profile "$PROFILE" \
      --risk_delta "$RDELTA" \
      --budget_fraction 0.60 \
      --consistency_check \
      --exp_name "${EPNAME}_trace5"
done

echo "[sbatch] P2 smoke triplet completed"
