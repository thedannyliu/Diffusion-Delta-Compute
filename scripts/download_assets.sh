#!/usr/bin/env bash
set -euo pipefail

# Roots (override via env)
DATA_ROOT=${DATA_ROOT:-/storage/ice1/2/9/eliu354/Projects/Diffusion-Delta-Compute/data}
MODELS_ROOT=${MODELS_ROOT:-/storage/ice1/2/9/eliu354/Projects/Diffusion-Delta-Compute/models}

mkdir -p "$DATA_ROOT" "$MODELS_ROOT"

echo "=== Downloading datasets locally ==="
# Wikitext-2 raw
if [ ! -d "$DATA_ROOT/wikitext-2-raw" ]; then
  wget -qO "$DATA_ROOT/wikitext-2-raw-v1.zip" https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
  unzip -q "$DATA_ROOT/wikitext-2-raw-v1.zip" -d "$DATA_ROOT"
  mv "$DATA_ROOT/wikitext-2-raw-v1" "$DATA_ROOT/wikitext-2-raw" || true
fi

# LAMBADA (OpenAI subset via Zenodo)
if [ ! -d "$DATA_ROOT/lambada" ]; then
  mkdir -p "$DATA_ROOT/lambada"
  wget -qO "$DATA_ROOT/lambada/lambada.tar.gz" https://zenodo.org/record/2630551/files/lambada-dataset.tar.gz || true
  tar -xzf "$DATA_ROOT/lambada/lambada.tar.gz" -C "$DATA_ROOT/lambada" || true
fi

# GSM8K (main)
if [ ! -d "$DATA_ROOT/gsm8k" ]; then
  mkdir -p "$DATA_ROOT/gsm8k"
  wget -qO "$DATA_ROOT/gsm8k/test.jsonl" https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl || true
fi

echo "=== Downloading D2F-Dream models (placeholder) ==="
if [ ! -d "$MODELS_ROOT/d2f-dream" ]; then
  git clone --depth=1 https://github.com/your_org/d2f-dream.git "$MODELS_ROOT/d2f-dream" || true
fi

# LoRA weights placeholder
mkdir -p "$MODELS_ROOT/d2f-lora"
echo "Place LoRA weights under $MODELS_ROOT/d2f-lora (manual for now)."

echo "Done."


