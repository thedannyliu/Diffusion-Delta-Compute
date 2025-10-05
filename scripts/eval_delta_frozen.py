#!/usr/bin/env python
"""Offline evaluator for δ_frozen using decision logs and Teacher outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


def load_outputs(path: str) -> Dict[int, Dict[str, List]]:
    outputs: Dict[int, Dict[str, List]] = {}
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            record = json.loads(line)
            prompt_id = int(record["prompt_id"])
            outputs[prompt_id] = {
                "tokens": record.get("tokens", []),
                "margins": record.get("margins", []),
            }
    return outputs


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    if trials == 0:
        return (0.0, 0.0)
    z = {
        0.80: 1.2816,
        0.90: 1.6449,
        0.95: 1.96,
        0.98: 2.3263,
        0.99: 2.5758,
    }.get(confidence, 1.96)
    phat = successes / trials
    denominator = 1 + (z * z) / trials
    centre = (phat + (z * z) / (2 * trials)) / denominator
    margin = z * math.sqrt((phat * (1 - phat) / trials) + (z * z) / (4 * trials * trials)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def evaluate_delta(
    decisions_path: str,
    teacher_outputs: Dict[int, Dict[str, List]],
    gate_outputs: Dict[int, Dict[str, List]],
    margin_threshold: float,
    sample_rate: float,
    max_samples: Optional[int],
    seed: int,
) -> Dict[str, object]:
    rng = random.Random(seed)
    total_entries = 0
    freeze_entries = 0
    sampled_freezes = 0
    unsafe_count = 0
    per_step_totals: Dict[int, int] = defaultdict(int)
    per_step_freeze: Dict[int, int] = defaultdict(int)
    per_step_unsafe: Dict[int, int] = defaultdict(int)
    coverage_missing_teacher = 0
    coverage_missing_gate = 0

    with open(decisions_path, "r", encoding="utf-8") as fp:
        for line in fp:
            record = json.loads(line)
            total_entries += 1
            step = int(record.get("step", 0))
            per_step_totals[step] += 1
            if record.get("decision") != "freeze":
                continue
            freeze_entries += 1
            per_step_freeze[step] += 1

            take = True
            if sample_rate < 1.0 and rng.random() > sample_rate:
                take = False
            if take and max_samples is not None and sampled_freezes >= max_samples:
                take = False
            if not take:
                continue

            sampled_freezes += 1
            prompt_id = int(record.get("prompt_id", -1))
            token_idx = int(record.get("token_idx", -1))

            teacher_record = teacher_outputs.get(prompt_id)
            gate_record = gate_outputs.get(prompt_id)
            if teacher_record is None or token_idx >= len(teacher_record.get("tokens", [])):
                coverage_missing_teacher += 1
                continue
            if gate_record is None or token_idx >= len(gate_record.get("tokens", [])):
                coverage_missing_gate += 1
                continue

            teacher_token = teacher_record["tokens"][token_idx]
            gate_token = gate_record["tokens"][token_idx]
            gate_margin = gate_record["margins"][token_idx] if token_idx < len(gate_record["margins"]) else 0.0
            mismatch = int(gate_token != teacher_token)
            low_margin = int(gate_margin < margin_threshold)
            watchdog_flag = int(bool(record.get("watchdog_triggered", False)))
            unsafe = mismatch or low_margin or watchdog_flag
            if unsafe:
                unsafe_count += 1
                per_step_unsafe[step] += 1

    delta_estimate = unsafe_count / sampled_freezes if sampled_freezes else 0.0
    ci_low, ci_high = wilson_interval(unsafe_count, sampled_freezes, 0.95)
    coverage = freeze_entries / total_entries if total_entries else 0.0

    per_step_summary = []
    for step in sorted(per_step_totals.keys()):
        total = per_step_totals[step]
        frozen = per_step_freeze.get(step, 0)
        unsafe_step = per_step_unsafe.get(step, 0)
        ratio = unsafe_step / frozen if frozen else 0.0
        ci_s, ci_e = wilson_interval(unsafe_step, frozen, 0.95) if frozen else (0.0, 0.0)
        per_step_summary.append(
            {
                "step": step,
                "token_total": total,
                "frozen": frozen,
                "unsafe": unsafe_step,
                "delta_estimate": ratio,
                "ci_low": ci_s,
                "ci_high": ci_e,
            }
        )

    return {
        "decisions_path": decisions_path,
        "total_tokens": total_entries,
        "frozen_tokens": freeze_entries,
        "sampled_frozen": sampled_freezes,
        "unsafe_count": unsafe_count,
        "delta_estimate": delta_estimate,
        "delta_ci_low": ci_low,
        "delta_ci_high": ci_high,
        "coverage": coverage,
        "coverage_missing_teacher": coverage_missing_teacher,
        "coverage_missing_gate": coverage_missing_gate,
        "per_step": per_step_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate δ_frozen from decision logs")
    parser.add_argument("--decisions", type=str, required=True, help="Path to decision JSONL log")
    parser.add_argument("--teacher_outputs", type=str, required=True, help="Path to teacher final outputs JSONL")
    parser.add_argument("--gate_outputs", type=str, required=True, help="Path to gating final outputs JSONL")
    parser.add_argument("--margin_threshold", type=float, default=0.07, help="Margin threshold for unsafe detection")
    parser.add_argument("--sample_rate", type=float, default=1.0, help="Optional subsampling rate for freeze tokens")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap number of freeze tokens to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Optional path to dump JSON summary")
    args = parser.parse_args()

    if not os.path.exists(args.decisions):
        raise FileNotFoundError(args.decisions)
    if not os.path.exists(args.teacher_outputs):
        raise FileNotFoundError(args.teacher_outputs)
    if not os.path.exists(args.gate_outputs):
        raise FileNotFoundError(args.gate_outputs)

    teacher_outputs = load_outputs(args.teacher_outputs)
    gate_outputs = load_outputs(args.gate_outputs)
    summary = evaluate_delta(
        decisions_path=args.decisions,
        teacher_outputs=teacher_outputs,
        gate_outputs=gate_outputs,
        margin_threshold=args.margin_threshold,
        sample_rate=args.sample_rate,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    print("δ_frozen estimate: {:.4f}".format(summary["delta_estimate"]))
    print("95% CI: [{:.4f}, {:.4f}]".format(summary["delta_ci_low"], summary["delta_ci_high"]))
    print("Coverage (freeze ratio): {:.4f}".format(summary["coverage"]))
    print("Evaluated freeze tokens: {} (sampled out of {})".format(summary["sampled_frozen"], summary["frozen_tokens"]))
    print("Unsafe tokens: {}".format(summary["unsafe_count"]))
    if summary.get("coverage_missing_teacher") or summary.get("coverage_missing_gate"):
        print(
            "Missing teacher outputs: {} | Missing gate outputs: {}".format(
                summary.get("coverage_missing_teacher", 0), summary.get("coverage_missing_gate", 0)
            )
        )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)


if __name__ == "__main__":
    main()
