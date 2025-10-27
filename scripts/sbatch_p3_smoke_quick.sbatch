#!/bin/bash
#SBATCH -J dcllm_p3_smoke_quick
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=60G
#SBATCH -t 01:45:00
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
mkdir -p logs models/gates

CONFIG_PATH=${CONFIG_PATH:-configs/p3_smoke.yaml}
ENGINE=${ENGINE:-d2f}
OUT_DIR=${OUT_DIR:-reports_quick}
DATA_ROOT=${DATA_ROOT:-}
WEIGHTS_OUT=${WEIGHTS_OUT:-models/gates/learned_gate_quick.npz}
P1_FEATURE_ROOT=${P1_FEATURE_ROOT:-$OUT_DIR}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-2}
TRAIN_BATCH=${TRAIN_BATCH:-512}
TRAIN_LR=${TRAIN_LR:-5e-4}

# Locate the newest teacher_features artifact for quick fine-tuning.
declare -a FEATURE_CANDIDATES=(
  "$P1_FEATURE_ROOT"
  "/home/hice1/eliu354/scratch/Projects/Diffusion-Delta-Compute/reports/P1"
)

FEATURE_PATH=""
for root in "${FEATURE_CANDIDATES[@]}"; do
  if [[ -d "$root" ]]; then
    mapfile -t FOUND < <(find "$root" -maxdepth 5 -type f -name 'teacher_features_*.npz' -print 2>/dev/null | LC_ALL=C sort)
    if (( ${#FOUND[@]} > 0 )); then
      FEATURE_PATH="${FOUND[-1]}"
      break
    fi
  fi
done

if [[ -n "$FEATURE_PATH" ]] && { [[ ! -f "$WEIGHTS_OUT" ]] || [[ -n "${FORCE_TRAIN_QUICK:-}" ]]; }; then
  echo "[sbatch][P3-quick] Training learned gate from features: $FEATURE_PATH"
  srun --exclusive -N1 -n1 \
    "$PYTHON" -m src.learned.train_gate \
      --features "$FEATURE_PATH" \
      --out "$WEIGHTS_OUT" \
      --epochs "$TRAIN_EPOCHS" \
      --batch_size "$TRAIN_BATCH" \
      --lr "$TRAIN_LR"
elif [[ ! -f "$WEIGHTS_OUT" ]]; then
  echo "[ERROR] Learned-gate weights not found at $WEIGHTS_OUT and no teacher_features_*.npz located; aborting." >&2
  exit 1
else
  echo "[sbatch][P3-quick] Reusing existing learned-gate weights: $WEIGHTS_OUT"
fi

declare -a DATASETS=(
  "gsm8k:5:gsm8k_learned_quick"
)

for spec in "${DATASETS[@]}"; do
  IFS=':' read -r TASK N EPNAME <<<"$spec"
  echo "[sbatch][P3-quick] Eval task=${TASK} n=${N}"
  srun --exclusive -N1 -n1 \
    "$PYTHON" -m src.run --mode learned_gate --config "$CONFIG_PATH" \
      --engine "$ENGINE" \
      --learned_gate_weights "$WEIGHTS_OUT" \
      --task "$TASK" \
      ${DATA_ROOT:+--data_root "$DATA_ROOT"} \
      --max_new_tokens 128 \
      --num_prompts "$N" \
      --out_dir "$OUT_DIR" \
      --exp_name "$EPNAME"
done

echo "[sbatch] P3 quick smoke completed"
