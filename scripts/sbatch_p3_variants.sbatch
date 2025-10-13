#!/bin/bash
#SBATCH -J dcllm_p3_variants
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
mkdir -p logs models/gates

CONFIG_PATH=${CONFIG_PATH:-configs/p3_smoke.yaml}
ENGINE=${ENGINE:-d2f}
WEIGHTS_OUT=${WEIGHTS_OUT:-models/gates/learned_gate_latest.npz}
# 指定 P1 teacher 輸出根目錄（供 features 使用）
P1_ROOT=${P1_ROOT:-/home/hice1/eliu354/scratch/Projects/Diffusion-Delta-Compute/reports/P1/teacher/d2f/gsm8k_smoke_20251006_002834}
# 可選資料根目錄（統一樣本/標註）
DATA_ROOT=${DATA_ROOT:-}

# 先訓練一次 learned gate 權重（使用 P1 的 features）
echo "[sbatch][P3-variants] Train learned gate from P1 features under: ${P1_ROOT}"
srun --exclusive -N1 -n1 bash -lc '
  set -euo pipefail
  PYTHON=${PYTHON:-python}
  if [[ ! -d "'"$P1_ROOT"'" ]]; then echo "[P3] ERROR: P1_ROOT='"$P1_ROOT"' not found." >&2; exit 1; fi
  mapfile -t F < <(find "'"$P1_ROOT"'/reports/artifacts/features" -maxdepth 1 -name "teacher_features_*.npz" -print 2>/dev/null | LC_ALL=C sort)
  if (( ${#F[@]} == 0 )); then echo "[P3] ERROR: No teacher_features_*.npz under '"$P1_ROOT"'." >&2; exit 1; fi
  echo "[learned_gate] Using features: ${F[-1]}"
  "$PYTHON" -m src.learned.train_gate --features "${F[-1]}" --out "'"$WEIGHTS_OUT"'" --epochs "${TRAIN_EPOCHS:-8}" --batch_size "${TRAIN_BATCH:-1024}" --lr "${TRAIN_LR:-1e-3}"
'

# 單一資料集（與 P1/P4/P5 對齊）
TASK=${TASK:-gsm8k}
N_PROMPTS=${N_PROMPTS:-200}

# Variants: NAME:THRESHOLD:FREEZE_K:MIN_CONSEC:BUDGET:RISK_DELTA
declare -a VARIANTS=(
  "cons:0.70:1:2:0.60:0.015"
  "bal:0.50:2:1:0.60:0.010"
  "aggr:0.30:3:1:0.50:0.008"
)

for spec in "${VARIANTS[@]}"; do
  IFS=':' read -r VNAME THRESH K MINC BUDGET RDELTA <<<"$spec"
  EXP_NAME="gsm8k_learned_${VNAME}"
  echo "[sbatch][P3-variants] Eval ${VNAME} task=${TASK} n=${N_PROMPTS} thr=${THRESH} K=${K} minc=${MINC} budget=${BUDGET} δ=${RDELTA}"
  srun --exclusive -N1 -n1 \
    "$PYTHON" -m src.run --mode learned_gate --config "$CONFIG_PATH" \
      --engine "$ENGINE" \
      --learned_gate_weights "$WEIGHTS_OUT" \
      --learned_gate_threshold "$THRESH" \
      --learned_gate_freeze_K "$K" \
      --learned_gate_min_consecutive "$MINC" \
      --task "$TASK" \
      ${DATA_ROOT:+--data_root "$DATA_ROOT"} \
      --max_new_tokens 256 \
      --num_prompts "$N_PROMPTS" \
      --risk_delta "$RDELTA" \
      --budget_fraction "$BUDGET" \
      --exp_name "$EXP_NAME"
done

echo "[sbatch] P3 variants completed"

