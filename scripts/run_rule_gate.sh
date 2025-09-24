#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=${1:-reports}

python -m src.run \
  --mode rule_gate \
  --engine mock \
  --out_dir "$OUT_DIR" \
  --num_prompts 8 \
  --max_new_tokens 64 \
  --num_steps 12 \
  --tau 0.95 \
  --rho 0.10 \
  --m 2 \
  --freeze_K 2 \
  --watchdog_min_cos 0.90 \
  --watchdog_max_kl 0.02 \
  --layer_recompute_M 2


