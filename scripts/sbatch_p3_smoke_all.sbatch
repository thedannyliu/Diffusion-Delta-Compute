#!/bin/bash
#SBATCH -J dcllm_p3_smoke_all
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 12:00:00
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
mkdir -p logs models/gates

CONFIG_PATH=${CONFIG_PATH:-configs/p3_smoke.yaml}
ENGINE=${ENGINE:-d2f}
WEIGHTS_OUT=${WEIGHTS_OUT:-models/gates/learned_gate_latest.npz}

# Train once from latest P1 features, then evaluate per dataset.
echo "[sbatch] P3 training learned gate from latest P1 features"
srun --exclusive -N1 -n1 bash -lc '
  PYTHON=${PYTHON:-python}
  PROJECT_ROOT=$(pwd)
  mapfile -t F < <(find "$PROJECT_ROOT/reports" -path "*teacher*features/teacher_features_*.npz" -print 2>/dev/null | sort)
  if [[ ${#F[@]} -eq 0 ]]; then echo "[P3] ERROR: No P1 features found." >&2; exit 1; fi
  "$PYTHON" -m src.learned.train_gate --features "${F[@]}" --out "'"$WEIGHTS_OUT"'" --epochs "${TRAIN_EPOCHS:-8}" --batch_size "${TRAIN_BATCH:-1024}" --lr "${TRAIN_LR:-1e-3}"
'

declare -a DATASETS=(
  "wikitext:2000:wikitext_learned"
  "lambada:2000:lambada_learned"
  "gsm8k:1500:gsm8k_learned"
)

for spec in "${DATASETS[@]}"; do
  IFS=':' read -r TASK N EPNAME <<<"$spec"
  echo "[sbatch][P3] Eval task=${TASK} n=${N}"
  srun --exclusive -N1 -n1 \
    bash scripts/run_p3_learned_gate.sh "" "$CONFIG_PATH" \
      --engine "$ENGINE" \
      --learned_gate_weights "$WEIGHTS_OUT" \
      --task "$TASK" \
      --num_prompts "$N" \
      --exp_name "$EPNAME"

  echo "[sbatch][P3] Trace5 task=${TASK}"
  srun --exclusive -N1 -n1 \
    bash scripts/run_p3_learned_gate.sh "" "$CONFIG_PATH" \
      --engine "$ENGINE" \
      --learned_gate_weights "$WEIGHTS_OUT" \
      --task "$TASK" \
      --num_prompts 5 \
      --consistency_check \
      --exp_name "${EPNAME}_trace5"
done

echo "[sbatch] P3 smoke triplet completed"

