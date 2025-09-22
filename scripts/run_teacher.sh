#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=${1:-reports}

python -m src.run \
  --mode teacher \
  --engine mock \
  --out_dir "$OUT_DIR" \
  --num_prompts 8 \
  --max_new_tokens 64 \
  --num_steps 12


