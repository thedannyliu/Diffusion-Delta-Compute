#!/usr/bin/env bash
set -euo pipefail

conda create -y -n dcllm python=3.10
source activate dcllm || conda activate dcllm

pip install torch==2.2.* --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate einops bitsandbytes
pip install sentencepiece tiktoken
pip install wandb evaluate scikit-learn matplotlib seaborn plotly
pip install thop fvcore torchprofile || true

echo "Environment ready."


