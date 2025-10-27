#!/bin/bash
#SBATCH -J dcllm_p3_smoke_all
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 02:30:00
#SBATCH -o logs/%x-%j.out

module load cuda/12.1 || true

unset PYTHONPATH

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

export PYTHON="/storage/ice1/2/9/eliu354/conda_envs/dcllm/bin/python"
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
FEATURE_OVERRIDE_DEFAULT=/home/hice1/2/9/eliu354/scratch/Projects/Diffusion-Delta-Compute/reports/P1/teacher/d2f/gsm8k_smoke_20251026_234728/reports/artifacts/features/teacher_features_20251026_234836.npz
FEATURE_OVERRIDE=${FEATURE_OVERRIDE:-$FEATURE_OVERRIDE_DEFAULT}
TRAIN_FEATURES=""
if [[ -f "$FEATURE_OVERRIDE" ]]; then
  TRAIN_FEATURES="$FEATURE_OVERRIDE"
  echo "[P3] Using explicit teacher features: $TRAIN_FEATURES"
else
  # 指定 P1 teacher 輸出根目錄（供 features 使用）
  P1_DEFAULT_REFERENCE=/home/hice1/eliu354/scratch/Projects/Diffusion-Delta-Compute/reports/P1/teacher/d2f/gsm8k_smoke_20251013_174729/reports/artifacts/features/teacher_features_20251013_174739.npz
  P1_REFERENCE=${P1_ROOT:-$P1_DEFAULT_REFERENCE}
  if [[ -f "$P1_REFERENCE" ]]; then
    # 允許直接給定 features 檔案，往上回推到 P1 執行根目錄
    P1_ROOT=$(dirname "$(dirname "$(dirname "$(dirname "$P1_REFERENCE")")")")
  else
    P1_ROOT="$P1_REFERENCE"
  fi
  if [[ -d "$P1_ROOT" ]]; then
    mapfile -t F < <(find "$P1_ROOT/reports/artifacts/features" -maxdepth 1 -name "teacher_features_*.npz" -print 2>/dev/null | LC_ALL=C sort)
    if (( ${#F[@]} > 0 )); then
      TRAIN_FEATURES="${F[-1]}"
      echo "[P3] Located fallback teacher features: $TRAIN_FEATURES"
    fi
  fi
fi
if [[ -z "$TRAIN_FEATURES" ]]; then
  echo "[P3] ERROR: 無法找到 teacher_features_*.npz；請設定 FEATURE_OVERRIDE 或 P1_ROOT。" >&2
  exit 1
fi
DATA_ROOT=${DATA_ROOT:-}

# Train once using P1 features, then evaluate per dataset.
echo "[sbatch] P3 training learned gate with features: ${TRAIN_FEATURES}"
srun --exclusive -N1 -n1 \
  "$PYTHON" -m src.learned.train_gate \
    --features "$TRAIN_FEATURES" \
    --out "$WEIGHTS_OUT" \
    --epochs "${TRAIN_EPOCHS:-8}" \
    --batch_size "${TRAIN_BATCH:-1024}" \
    --lr "${TRAIN_LR:-1e-3}"

declare -a DATASETS=(
  "gsm8k:10:gsm8k_learned"
)

for spec in "${DATASETS[@]}"; do
  IFS=':' read -r TASK N EPNAME <<<"$spec"
  echo "[sbatch][P3] Eval task=${TASK} n=${N}"
  srun --exclusive -N1 -n1 \
    "$PYTHON" -m src.run --mode learned_gate --config "$CONFIG_PATH" \
      --engine "$ENGINE" \
      --learned_gate_weights "$WEIGHTS_OUT" \
      --task "$TASK" \
      ${DATA_ROOT:+--data_root "$DATA_ROOT"} \
      --max_new_tokens 256 \
      --num_prompts "$N" \
      --exp_name "$EPNAME"

  echo "[sbatch][P3] Trace5 task=${TASK}"
  srun --exclusive -N1 -n1 \
    "$PYTHON" -m src.run --mode learned_gate --config "$CONFIG_PATH" \
      --engine "$ENGINE" \
      --learned_gate_weights "$WEIGHTS_OUT" \
      --task "$TASK" \
      ${DATA_ROOT:+--data_root "$DATA_ROOT"} \
      --max_new_tokens 256 \
      --num_prompts "$N" \
      --consistency_check \
      --exp_name "${EPNAME}_trace5"
done

echo "[sbatch] P3 smoke triplet completed"
