#!/usr/bin/env bash
set -euo pipefail

ENGINE=${1:-mock}        # mock | d2f
PHASE=${2:-P1}           # P1 | P2 | P3
OUT_ROOT=${3:-/storage/ice1/2/9/eliu354/Projects/Diffusion-Delta-Compute/reports}

# Example: run teacher (P1)
if [ "$PHASE" = "P1" ]; then
  python -m src.run \
    --mode teacher \
    --engine "$ENGINE" \
    --out_dir "$OUT_ROOT" \
    --num_prompts 64 \
    --max_new_tokens 128 \
    --num_steps 12
  exit 0
fi

# Example: run rule gate (P2)
if [ "$PHASE" = "P2" ]; then
  python -m src.run \
    --mode rule_gate \
    --engine "$ENGINE" \
    --out_dir "$OUT_ROOT" \
    --num_prompts 64 \
    --max_new_tokens 128 \
    --num_steps 12 \
    --tau 0.95 --rho 0.10 --m 2 --freeze_K 2 \
    --watchdog_min_cos 0.90 --watchdog_max_kl 0.02 \
    --consistency_check
  exit 0
fi

echo "Unknown phase: $PHASE"
exit 1


