#!/bin/bash
#SBATCH -J dcllm_p3_learned
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out

module load cuda/12.1 anaconda
source activate dcllm || conda activate dcllm
export WANDB_MODE=${WANDB_MODE:-offline}
mkdir -p logs

FEATURE_SOURCE=${FEATURE_SOURCE:-}
CONFIG_PATH=${CONFIG_PATH:-configs/p3_smoke.yaml}
ENGINE=${ENGINE:-d2f}
OUT_DIR=${OUT_DIR:-}
WEIGHTS_OUT=${WEIGHTS_OUT:-models/gates/learned_gate_latest.npz}

if [[ -n "$FEATURE_SOURCE" ]]; then
  echo "[sbatch] Using feature glob: $FEATURE_SOURCE"
else
  echo "[sbatch] Using latest P1 features (auto-discovery)"
fi

RUN_ARGS=()
if [[ -n "$FEATURE_SOURCE" ]]; then
  RUN_ARGS+=("$FEATURE_SOURCE")
fi
if [[ -n "$CONFIG_PATH" ]]; then
  RUN_ARGS+=("$CONFIG_PATH")
fi

srun FEATURE_SOURCE="$FEATURE_SOURCE" CONFIG_PATH="$CONFIG_PATH" ENGINE="$ENGINE" OUT_DIR="$OUT_DIR" WEIGHTS_OUT="$WEIGHTS_OUT" bash scripts/run_p3_learned_gate.sh "${RUN_ARGS[@]}" "$@"
