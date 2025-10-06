#!/bin/bash
#SBATCH -J dcllm_p3_smoke_all
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

# Train once using P1 features, then evaluate per dataset.
echo "[sbatch] P3 training learned gate from P1 features under: ${P1_ROOT}"
srun --exclusive -N1 -n1 bash -lc '
  set -euo pipefail
  PYTHON=${PYTHON:-python}
  if [[ ! -d "'"$P1_ROOT"'" ]]; then echo "[P3] ERROR: P1_ROOT='"$P1_ROOT"' not found." >&2; exit 1; fi
  mapfile -t F < <(find "'"$P1_ROOT"'/reports/artifacts/features" -maxdepth 1 -name "teacher_features_*.npz" -print 2>/dev/null | LC_ALL=C sort)
  if (( ${#F[@]} == 0 )); then echo "[P3] ERROR: No teacher_features_*.npz under '"$P1_ROOT"'." >&2; exit 1; fi
  echo "[learned_gate] Using features: ${F[-1]}"
  "$PYTHON" -m src.learned.train_gate --features "${F[-1]}" --out "'"$WEIGHTS_OUT"'" --epochs "${TRAIN_EPOCHS:-8}" --batch_size "${TRAIN_BATCH:-1024}" --lr "${TRAIN_LR:-1e-3}"
'

declare -a DATASETS=(
  "gsm8k:200:gsm8k_learned"
)

for spec in "${DATASETS[@]}"; do
  IFS=':' read -r TASK N EPNAME <<<"$spec"
  echo "[sbatch][P3] Eval task=${TASK} n=${N}"
  srun --exclusive -N1 -n1 \
    "$PYTHON" -m src.run --mode learned_gate --config "$CONFIG_PATH" \
      --engine "$ENGINE" \
      --learned_gate_weights "$WEIGHTS_OUT" \
      --task "$TASK" \
      --num_prompts "$N" \
      --exp_name "$EPNAME"

  echo "[sbatch][P3] Trace5 task=${TASK}"
  srun --exclusive -N1 -n1 \
    "$PYTHON" -m src.run --mode learned_gate --config "$CONFIG_PATH" \
      --engine "$ENGINE" \
      --learned_gate_weights "$WEIGHTS_OUT" \
      --task "$TASK" \
      --num_prompts 200 \
      --consistency_check \
      --exp_name "${EPNAME}_trace5"
done

echo "[sbatch] P3 smoke triplet completed"
