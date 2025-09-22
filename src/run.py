import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np

from src.utils.logging import init_run_logging
from src.utils.timer import Timer
from src.engine.mock_engine import MockDiffusionEngine
from src.engine.d2f_engine import D2FDreamEngine
from src.probes.metrics import (
    cosine_similarity_tokens,
    delta_l2_ratio_tokens,
    kl_divergence_tokens,
    logits_entropy_and_margin,
)
from src.gating.rule_gate import RuleGate, RuleGateConfig, StepGateResult
from src.utils.profile import query_gpu_utilization
from src.utils.config import load_yaml, ensure_output_dirs, save_manifest
import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delta-Compute dLLM PoC runner")
    parser.add_argument("--mode", type=str, choices=["teacher", "rule_gate"], required=True)
    parser.add_argument("--engine", type=str, default="mock", choices=["mock", "d2f"], help="Engine backend")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for runs")
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=12)
    parser.add_argument("--consistency_check", action="store_true", help="For rule_gate, run a full-compute baseline to check final token consistency")
    parser.add_argument("--config", type=str, default=None, help="YAML config with paths and params")
    parser.add_argument("--phase", type=str, default="P1", help="Experiment phase: P1|P2|P3")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--models_root", type=str, default=None)
    parser.add_argument("--use_d2f_lora", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    # Rule gate thresholds
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--rho", type=float, default=0.10)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--freeze_K", type=int, default=2)
    parser.add_argument("--watchdog_min_cos", type=float, default=0.90)
    parser.add_argument("--watchdog_max_kl", type=float, default=0.02)
    parser.add_argument("--layer_recompute_M", type=int, default=2)
    return parser.parse_args()


def synthetic_prompts(num_prompts: int) -> List[str]:
    base_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A wizard's job is to vex chumps quickly in fog.",
        "Pack my box with five dozen liquor jugs.",
        "Sphinx of black quartz, judge my vow.",
        "How vexingly quick daft zebras jump!",
    ]
    prompts: List[str] = []
    for i in range(num_prompts):
        prompts.append(base_sentences[i % len(base_sentences)])
    return prompts


def ensure_dirs(out_dir: str) -> Dict[str, str]:
    figures_dir = os.path.join(out_dir, "figures")
    runs_dir = os.path.join(out_dir, "runs")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    return {"figures": figures_dir, "runs": runs_dir}


def save_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def run_teacher(engine: MockDiffusionEngine, prompts: List[str], out_dirs: Dict[str, str], num_steps: int) -> None:
    from src.viz.plots import save_heatmap, save_hist

    aggregates: List[Dict] = []
    cos_means: List[List[float]] = []
    dl2_means: List[List[float]] = []
    cos_hist_samples: List[float] = []

    per_seq_latency_ms: List[float] = []
    total_start = time.time()

    for prompt in prompts:
        state = engine.encode_prompt(prompt)
        prev_hidden: Optional[List[np.ndarray]] = None
        prev_logits: Optional[np.ndarray] = None
        step_cos_means: List[float] = []
        step_dl2_means: List[float] = []

        seq_start = time.time()
        for t in range(num_steps):
            out = engine.step(state, t)
            if prev_hidden is not None and prev_logits is not None:
                cos = cosine_similarity_tokens(out.hidden_by_layer, prev_hidden)
                dl2 = delta_l2_ratio_tokens(out.hidden_by_layer, prev_hidden)
                kl = kl_divergence_tokens(out.logits, prev_logits)
                ent, margin = logits_entropy_and_margin(out.logits)
                step_cos_means.append(float(np.mean(cos)))
                step_dl2_means.append(float(np.mean(dl2)))
                cos_hist_samples.extend(cos.flatten().tolist())
                aggregates.append(
                    {
                        "mode": "teacher",
                        "prompt_len": len(prompt),
                        "step": t,
                        "cos_mean": float(np.mean(cos)),
                        "dl2_mean": float(np.mean(dl2)),
                        "kl_mean": float(np.mean(kl)),
                        "entropy_mean": float(np.mean(ent)),
                        "margin_mean": float(np.mean(margin)),
                    }
                )
            prev_hidden = [h.copy() for h in out.hidden_by_layer]
            prev_logits = out.logits.copy()

        cos_means.append(step_cos_means)
        dl2_means.append(step_dl2_means)
        per_seq_latency_ms.append((time.time() - seq_start) * 1000.0)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_jsonl(os.path.join(out_dirs["runs"], f"teacher_{timestamp}.jsonl"), aggregates)

    if len(cos_means) > 0 and len(cos_means[0]) > 0:
        arr = np.array(cos_means).T  # steps x prompts
        save_heatmap(arr, os.path.join(out_dirs["figures"], f"teacher_cosine_heatmap_{timestamp}.png"), title="Cosine mean per step (prompts as columns)")
    if len(dl2_means) > 0 and len(dl2_means[0]) > 0:
        arr = np.array(dl2_means).T
        save_heatmap(arr, os.path.join(out_dirs["figures"], f"teacher_dl2_heatmap_{timestamp}.png"), title="ΔL2 mean per step")
    if len(cos_hist_samples) > 0:
        save_hist(np.array(cos_hist_samples), os.path.join(out_dirs["figures"], f"teacher_cosine_hist_{timestamp}.png"), title="Cosine similarity histogram")

    total_time_s = max(1e-6, time.time() - total_start)
    gpu = query_gpu_utilization() or {}
    summary = {
        "mode": "teacher",
        "num_prompts": len(prompts),
        "num_steps": num_steps,
        "latency_ms_mean": float(np.mean(per_seq_latency_ms) if per_seq_latency_ms else 0.0),
        "latency_ms_p50": float(np.percentile(per_seq_latency_ms, 50) if per_seq_latency_ms else 0.0),
        "latency_ms_p90": float(np.percentile(per_seq_latency_ms, 90) if per_seq_latency_ms else 0.0),
        "throughput_seq_per_s": float(len(prompts) / total_time_s),
        **{f"gpu_{k}": v for k, v in gpu.items()},
    }
    with open(os.path.join(out_dirs["runs"], f"teacher_summary_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def run_rule_gate(
    engine: MockDiffusionEngine,
    prompts: List[str],
    out_dirs: Dict[str, str],
    num_steps: int,
    cfg: RuleGateConfig,
    consistency_check: bool = False,
) -> None:
    from src.viz.plots import save_hist

    aggregates: List[Dict] = []
    savings_ratios: List[float] = []

    gate = RuleGate(cfg)

    per_seq_latency_ms: List[float] = []
    final_consistency: List[float] = []
    total_start = time.time()

    for prompt in prompts:
        state = engine.encode_prompt(prompt)
        prev_hidden: Optional[List[np.ndarray]] = None
        prev_logits: Optional[np.ndarray] = None

        total_tokens_computed = 0
        total_tokens_possible = 0

        seq_start = time.time()
        for t in range(num_steps):
            if prev_hidden is None or prev_logits is None:
                out = engine.step(state, t)
                prev_hidden = [h.copy() for h in out.hidden_by_layer]
                prev_logits = out.logits.copy()
                total_tokens_computed += out.logits.shape[0]
                total_tokens_possible += out.logits.shape[0]
                continue

            # Get compute mask based on current freeze cooldowns
            compute_mask = gate.get_compute_mask(prev_logits.shape[0])
            out = engine.step(state, t, compute_mask=compute_mask)

            # Watchdog check using actual new outputs vs prev
            cos_now = cosine_similarity_tokens(out.hidden_by_layer, prev_hidden)
            kl_now = kl_divergence_tokens(out.logits, prev_logits)
            if np.any((cos_now < cfg.watchdog_min_cos) | (kl_now > cfg.watchdog_max_kl)):
                out = engine.step(state, t, compute_mask=None)

            # Accounting
            total_tokens_computed += int(np.sum(compute_mask) if compute_mask is not None else out.logits.shape[0])
            total_tokens_possible += out.logits.shape[0]

            # Update prev
            prev_hidden = [h.copy() for h in out.hidden_by_layer]
            prev_logits = out.logits.copy()

            # Compute features using actual new vs prev (after update) for next-step gating update
            feats_cos = cosine_similarity_tokens(prev_hidden, prev_hidden)  # placeholder ones
            feats_dl2 = np.zeros_like(feats_cos)
            feats_kl = np.zeros_like(feats_cos)
            feats_entropy, feats_margin = logits_entropy_and_margin(prev_logits)
            gate.update(
                feats_cos=feats_cos,
                feats_dl2=feats_dl2,
                feats_kl=feats_kl,
                entropy=feats_entropy,
                margin=feats_margin,
                num_tokens=prev_logits.shape[0],
            )

        per_seq_latency_ms.append((time.time() - seq_start) * 1000.0)

        savings = 1.0 - (total_tokens_computed / max(1, total_tokens_possible))
        savings_ratios.append(float(savings))
        aggregates.append({"mode": "rule_gate", "prompt_len": len(prompt), "savings": float(savings)})

        if consistency_check:
            # Run full-compute baseline for the same prompt and compare final token argmax
            baseline_state = engine.encode_prompt(prompt)
            baseline_prev_logits: Optional[np.ndarray] = None
            for t in range(num_steps):
                baseline_out = engine.step(baseline_state, t, compute_mask=None)
                baseline_prev_logits = baseline_out.logits
            if baseline_prev_logits is not None and prev_logits is not None:
                pred_a = np.argmax(prev_logits, axis=-1)
                pred_b = np.argmax(baseline_prev_logits, axis=-1)
                consistency = float(np.mean((pred_a == pred_b).astype(np.float32)))
                final_consistency.append(consistency)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_jsonl(os.path.join(out_dirs["runs"], f"rule_gate_{timestamp}.jsonl"), aggregates)
    if len(savings_ratios) > 0:
        save_hist(np.array(savings_ratios), os.path.join(out_dirs["figures"], f"rule_gate_savings_hist_{timestamp}.png"), title="Token compute savings ratio")

    total_time_s = max(1e-6, time.time() - total_start)
    gpu = query_gpu_utilization() or {}
    summary = {
        "mode": "rule_gate",
        "num_prompts": len(prompts),
        "num_steps": num_steps,
        "latency_ms_mean": float(np.mean(per_seq_latency_ms) if per_seq_latency_ms else 0.0),
        "latency_ms_p50": float(np.percentile(per_seq_latency_ms, 50) if per_seq_latency_ms else 0.0),
        "latency_ms_p90": float(np.percentile(per_seq_latency_ms, 90) if per_seq_latency_ms else 0.0),
        "throughput_seq_per_s": float(len(prompts) / total_time_s),
        "skip_ratio_mean": float(np.mean(savings_ratios) if savings_ratios else 0.0),
        "final_token_consistency_mean": float(np.mean(final_consistency) if final_consistency else -1.0),
        **{f"gpu_{k}": v for k, v in gpu.items()},
    }
    with open(os.path.join(out_dirs["runs"], f"rule_gate_summary_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    phase = args.phase or cfg.get("phase", "P1")
    engine_name = args.engine or cfg.get("engine", "mock")
    data_root = args.data_root or cfg.get("data_root")
    models_root = args.models_root or cfg.get("models_root")
    outputs_root = args.out_dir or cfg.get("outputs_root", args.out_dir)
    exp_name = args.exp_name or cfg.get("exp_name")

    run_dir = ensure_output_dirs(outputs_root, phase=phase, mode=args.mode, engine=engine_name, exp_name=exp_name)
    init_run_logging(run_dir, f"{args.mode}-{engine_name}")
    save_manifest(run_dir, {
        "phase": phase,
        "mode": args.mode,
        "engine": engine_name,
        "args": vars(args),
        "config": cfg,
    })

    wandb.init(project=os.environ.get("WANDB_PROJECT", "dcllm-delta"), name=os.path.basename(run_dir), dir=run_dir)

    if engine_name == "mock":
        engine = MockDiffusionEngine(
            vocab_size=1000,
            num_layers=4,
            hidden_size=64,
            sequence_length=args.max_new_tokens,
            num_steps=args.num_steps,
            rng_seed=42,
        )
    elif engine_name == "d2f":
        engine = D2FDreamEngine(
            model_name="d2f-small",
            device="cuda",
            model_path=args.model_path or cfg.get("model_path"),
            lora_path=args.lora_path or cfg.get("lora_path"),
            use_lora=bool(args.use_d2f_lora or cfg.get("use_d2f_lora", False)),
        )
    else:
        raise ValueError(f"Unsupported engine: {engine_name}")

    prompts = synthetic_prompts(args.num_prompts)
    out_dirs = {"figures": os.path.join(run_dir, "figures"), "runs": os.path.join(run_dir, "runs")}

    with Timer("run"):
        if args.mode == "teacher":
            run_teacher(engine, prompts, out_dirs, args.num_steps)
        elif args.mode == "rule_gate":
            cfg = RuleGateConfig(
                cosine_tau=args.tau,
                delta_l2_rho=args.rho,
                consecutive_m=args.m,
                freeze_K=args.freeze_K,
                watchdog_min_cos=args.watchdog_min_cos,
                watchdog_max_kl=args.watchdog_max_kl,
                layer_recompute_M=args.layer_recompute_M,
            )
            run_rule_gate(engine, prompts, out_dirs, args.num_steps, cfg, consistency_check=args.consistency_check)
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()


