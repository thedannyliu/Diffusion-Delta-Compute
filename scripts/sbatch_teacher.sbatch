#!/bin/bash
#SBATCH -J dcllm_teacher
#SBATCH -A <ACCOUNT>
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out

module load cuda/12.1 anaconda
source activate dcllm || conda activate dcllm
export WANDB_MODE=offline

srun bash scripts/run_teacher.sh reports


