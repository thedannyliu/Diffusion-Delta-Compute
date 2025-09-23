#!/bin/bash
#SBATCH -J dcllm_p2_rule
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

CONFIG_PATH=${CONFIG_PATH:-configs/p2_smoke.yaml}
ENGINE=${ENGINE:-d2f}
OUT_DIR=${OUT_DIR:-}

echo "[sbatch] P2 rule gate run using config=${CONFIG_PATH} engine=${ENGINE}"
srun ENGINE="$ENGINE" OUT_DIR="$OUT_DIR" bash scripts/run_p2_rule_gate.sh "$CONFIG_PATH" "$@"
