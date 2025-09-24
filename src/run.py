import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.logging import init_run_logging
from src.utils.timer import Timer
from src.engine.base_engine import BaseEngine
from src.engine.mock_engine import MockDiffusionEngine

try:  # Optional dependency, only needed when using the d2f backend
    from src.engine.d2f_engine import D2FDreamEngine
except Exception:  # pragma: no cover - lazy import guard
    D2FDreamEngine = None
from src.probes.metrics import (
    cosine_similarity_tokens,
    delta_l2_ratio_tokens,
    kl_divergence_tokens,
    logits_entropy_and_margin,
)
from src.gating.rule_gate import RuleGate, RuleGateConfig, StepGateResult
from src.gating.learned_gate import LearnedGate, LearnedGateConfig, LearnedGateWeights
from src.eval.tasks import load_wikitext2, load_lambada_openai, load_gsm8k_tiny
from src.utils.profile import query_gpu_utilization
from src.utils.config import load_yaml, ensure_output_dirs, save_manifest

try:  # Optional dependency; scripts can run without W&B
    import wandb  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    wandb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delta-Compute dLLM PoC runner")
    parser.add_argument("--mode", type=str, choices=["teacher", "rule_gate", "learned_gate"], required=True)
    parser.add_argument("--engine", type=str, default=None, choices=["mock", "d2f"], help="Engine backend")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for runs")
    parser.add_argument("--num_prompts", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--consistency_check", action="store_true", help="Run full-compute baseline to check final token consistency")
    parser.add_argument("--config", type=str, default=None, help="YAML config with paths and params")
    parser.add_argument("--phase", type=str, default=None, help="Experiment phase: P1|P2|P3")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--models_root", type=str, default=None)
    parser.add_argument("--outputs_root", type=str, default=None)
    parser.add_argument("--use_d2f_lora", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--task", type=str, default=None, choices=["synthetic", "wikitext", "lambada", "gsm8k"], help="Prompt source")
    parser.add_argument("--dump_features_dir", type=str, default=None, help="Directory to dump token-level features for training the learned gate")
    parser.add_argument("--label_cos_tau", type=float, default=None, help="Cosine threshold for positive labels when dumping features")
    parser.add_argument("--label_dl2_rho", type=float, default=None, help="ΔL2 threshold for positive labels when dumping features")
    parser.add_argument("--label_kl_max", type=float, default=None, help="KL threshold for positive labels when dumping features")
    parser.add_argument("--learned_gate_weights", type=str, default=None, help="Path to learned gate weight file (.npz)")
    parser.add_argument("--learned_gate_threshold", type=float, default=None, help="Probability threshold for the learned gate")
    parser.add_argument("--learned_gate_freeze_K", type=int, default=None, help="Freeze horizon for learned gate")
    parser.add_argument("--learned_gate_min_consecutive", type=int, default=None, help="Consecutive stable steps before freezing in learned gate")
    # Rule gate thresholds
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--freeze_K", type=int, default=None)
    parser.add_argument("--watchdog_min_cos", type=float, default=None)
    parser.add_argument("--watchdog_max_kl", type=float, default=None)
    parser.add_argument("--layer_recompute_M", type=int, default=None)
    parser.add_argument("--consistency_full_compute", action="store_true", help="Run full-compute baseline and compare outputs; write difference report")
    return parser.parse_args()


def coalesce(*values):
    for value in values:
        if value is not None:
            return value
    return None


def load_default_paths(custom_path: Optional[str] = None) -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    candidate = Path(custom_path) if custom_path else (repo_root / "configs" / "paths.yaml")
    if candidate.exists():
        return load_yaml(str(candidate))
    return {}


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


def load_prompts(num_prompts: int, task: str, data_root: Optional[str]) -> List[str]:
    if task == "synthetic":
        return synthetic_prompts(num_prompts)
    if task == "wikitext":
        return load_wikitext2(n_eval=num_prompts, data_root=data_root)
    if task == "lambada":
        batch = load_lambada_openai(n_eval=num_prompts, data_root=data_root)
        return batch.texts
    if task == "gsm8k":
        batch = load_gsm8k_tiny(n_eval=num_prompts, data_root=data_root)
        return batch.texts
    raise ValueError(f"Unsupported task: {task}")


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


FEATURE_NAMES = ("cosine", "delta_l2", "kl", "entropy", "margin", "step_frac")


def compute_token_observables(
    current_hidden: List[np.ndarray],
    prev_hidden: List[np.ndarray],
    current_logits: np.ndarray,
    prev_logits: np.ndarray,
    step_index: int,
    num_steps: int,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cos = cosine_similarity_tokens(current_hidden, prev_hidden)
    dl2 = delta_l2_ratio_tokens(current_hidden, prev_hidden)
    kl = kl_divergence_tokens(current_logits, prev_logits)
    entropy, margin = logits_entropy_and_margin(current_logits)
    if valid_mask is not None:
        cos = cos[valid_mask]
        dl2 = dl2[valid_mask]
        kl = kl[valid_mask]
        entropy = entropy[valid_mask]
        margin = margin[valid_mask]
    denom = max(1, num_steps - 1)
    step_frac = np.full_like(cos, fill_value=float(step_index) / float(denom), dtype=np.float32)
    return cos, dl2, kl, entropy, margin, step_frac


def run_teacher(
    engine: BaseEngine,
    prompts: List[str],
    out_dirs: Dict[str, str],
    num_steps: int,
    dump_features_dir: Optional[str] = None,
    label_thresholds: Optional[Dict[str, float]] = None,
) -> Optional[str]:
    from src.viz.plots import save_heatmap, save_hist

    aggregates: List[Dict] = []
    cos_means: List[List[float]] = []
    dl2_means: List[List[float]] = []
    kl_means: List[List[float]] = []
    ent_means: List[List[float]] = []
    cos_hist_samples: List[float] = []

    feature_blocks: List[np.ndarray] = []
    feature_labels: List[np.ndarray] = []
    feature_meta: List[np.ndarray] = []

    per_seq_latency_ms: List[float] = []
    total_start = time.time()

    label_cfg = label_thresholds or {}
    cos_tau = float(label_cfg.get("cosine_tau", 0.97))
    dl2_rho = float(label_cfg.get("delta_l2_rho", 0.05))
    kl_max = float(label_cfg.get("kl_max", 0.01))
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for prompt_idx, prompt in enumerate(prompts):
        state = engine.encode_prompt(prompt)
        prev_hidden: Optional[List[np.ndarray]] = None
        prev_logits: Optional[np.ndarray] = None
        prev_aux: Optional[Dict] = None
        step_cos_means: List[float] = []
        step_dl2_means: List[float] = []
        step_kl_means: List[float] = []
        step_ent_means: List[float] = []
        # Per-layer accumulation for mean and IQR
        layer_step_values_cos: Dict[Tuple[int, int], np.ndarray] = {}
        layer_step_values_dl2: Dict[Tuple[int, int], np.ndarray] = {}

        seq_start = time.time()
        for t in range(num_steps):
            out = engine.step(state, t)
            if prev_hidden is not None and prev_logits is not None:
                # Build valid mask (exclude padding and special tokens)
                valid_mask: Optional[np.ndarray] = None
                if isinstance(out.aux, dict):
                    ids = out.aux.get("input_ids")
                    special = out.aux.get("special_tokens_mask")
                    pad_id = out.aux.get("pad_token_id", -1)
                    if ids is not None:
                        ids = np.array(ids, dtype=np.int64)
                        is_pad = (ids == pad_id) if pad_id != -1 else np.zeros_like(ids, dtype=bool)
                        is_special = np.array(special, dtype=bool) if special is not None else np.zeros_like(ids, dtype=bool)
                        valid_mask = ~(is_pad | is_special)
                cos, dl2, kl, ent, margin, step_frac = compute_token_observables(
                    out.hidden_by_layer,
                    prev_hidden,
                    out.logits,
                    prev_logits,
                    t,
                    num_steps,
                    valid_mask=valid_mask,
                )
                step_cos_means.append(float(np.mean(cos)))
                step_dl2_means.append(float(np.mean(dl2)))
                step_kl_means.append(float(np.mean(kl)))
                step_ent_means.append(float(np.mean(ent)))
                cos_hist_samples.extend(cos.flatten().tolist())
                # Save per-layer token vectors for IQR
                for li, (h_now, h_prev) in enumerate(zip(out.hidden_by_layer, prev_hidden)):
                    # Compute per-token layer-wise metrics (masked)
                    cos_layer = cosine_similarity_tokens([h_now], [h_prev])
                    dl2_layer = delta_l2_ratio_tokens([h_now], [h_prev])
                    if valid_mask is not None:
                        cos_layer = cos_layer[valid_mask]
                        dl2_layer = dl2_layer[valid_mask]
                    layer_step_values_cos[(li, t)] = cos_layer
                    layer_step_values_dl2[(li, t)] = dl2_layer
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

                if dump_features_dir:
                    features = np.stack([cos, dl2, kl, ent, margin, step_frac], axis=-1).astype(np.float32)
                    labels = ((cos >= cos_tau) & (dl2 <= dl2_rho) & (kl <= kl_max)).astype(np.float32)
                    feature_blocks.append(features.reshape(-1, features.shape[-1]))
                    feature_labels.append(labels.reshape(-1))
                    shape = labels.reshape(-1).shape
                    meta = np.stack(
                        [
                            np.full(shape, prompt_idx, dtype=np.int32),
                            np.full(shape, t, dtype=np.int32),
                        ],
                        axis=-1,
                    )
                    feature_meta.append(meta)
            # Snapshot prev strictly after metrics (prevent view overwrite)
            prev_hidden = [h.copy() for h in out.hidden_by_layer]
            prev_logits = out.logits.copy()
            prev_aux = out.aux if isinstance(out.aux, dict) else None

        cos_means.append(step_cos_means)
        dl2_means.append(step_dl2_means)
        kl_means.append(step_kl_means)
        ent_means.append(step_ent_means)
        per_seq_latency_ms.append((time.time() - seq_start) * 1000.0)

    save_jsonl(os.path.join(out_dirs["runs"], f"teacher_{timestamp}.jsonl"), aggregates)
    # Export layer×step mean and IQR to CSV
    teacher_csv = os.path.join(out_dirs["runs"], f"teacher_aggregates_{timestamp}.csv")
    with open(teacher_csv, "w", encoding="utf-8") as f:
        f.write("layer,step,metric,mean,q25,q75\n")
        # infer num_layers, num_steps from collected keys
        cos_keys = list(layer_step_values_cos.keys())
        steps = sorted(set(k[1] for k in cos_keys))
        layers = sorted(set(k[0] for k in cos_keys))
        for li in layers:
            for t in steps:
                if (li, t) in layer_step_values_cos:
                    v = layer_step_values_cos[(li, t)]
                    f.write(f"{li},{t},cos,{np.mean(v):.6f},{np.percentile(v,25):.6f},{np.percentile(v,75):.6f}\n")
                if (li, t) in layer_step_values_dl2:
                    v = layer_step_values_dl2[(li, t)]
                    f.write(f"{li},{t},dl2,{np.mean(v):.6f},{np.percentile(v,25):.6f},{np.percentile(v,75):.6f}\n")

    if len(cos_means) > 0 and len(cos_means[0]) > 0:
        arr = np.array(cos_means).T  # steps x prompts
        save_heatmap(arr, os.path.join(out_dirs["figures"], f"teacher_cosine_heatmap_{timestamp}.png"), title="Cosine mean per step (prompts as columns)")
    if len(dl2_means) > 0 and len(dl2_means[0]) > 0:
        arr = np.array(dl2_means).T
        save_heatmap(arr, os.path.join(out_dirs["figures"], f"teacher_dl2_heatmap_{timestamp}.png"), title="ΔL2 mean per step")
    # Export KL/entropy step curves to CSV for further plotting
    if kl_means:
        kl_csv = os.path.join(out_dirs["runs"], f"teacher_kl_entropy_{timestamp}.csv")
        steps = list(range(len(kl_means[0]))) if kl_means[0] else []
        with open(kl_csv, "w", encoding="utf-8") as f:
            f.write("step,kl_mean,entropy_mean\n")
            for s in steps:
                km = float(np.mean([row[s] for row in kl_means if len(row) > s]))
                em = float(np.mean([row[s] for row in ent_means if len(row) > s]))
                f.write(f"{s},{km:.6f},{em:.6f}\n")
    # Export layer bands (L1-3, L4-6, L7+) IQR/mean
    if layer_step_values_cos:
        bands_csv = os.path.join(out_dirs["runs"], f"teacher_layer_bands_{timestamp}.csv")
        with open(bands_csv, "w", encoding="utf-8") as f:
            f.write("band,step,metric,mean,q25,q75\n")
            cos_keys = list(layer_step_values_cos.keys())
            steps = sorted(set(k[1] for k in cos_keys))
            layers = sorted(set(k[0] for k in cos_keys))
            # Define bands
            def band_of(li: int) -> str:
                if li <= 2:
                    return "L1-3"
                if li <= 5:
                    return "L4-6"
                return "L7+"
            for t in steps:
                band_to_vals_cos: Dict[str, List[float]] = {"L1-3": [], "L4-6": [], "L7+": []}
                band_to_vals_dl2: Dict[str, List[float]] = {"L1-3": [], "L4-6": [], "L7+": []}
                for li in layers:
                    b = band_of(li)
                    if (li, t) in layer_step_values_cos:
                        band_to_vals_cos[b].extend(layer_step_values_cos[(li, t)].tolist())
                    if (li, t) in layer_step_values_dl2:
                        band_to_vals_dl2[b].extend(layer_step_values_dl2[(li, t)].tolist())
                for b in ["L1-3", "L4-6", "L7+"]:
                    if band_to_vals_cos[b]:
                        v = np.array(band_to_vals_cos[b], dtype=np.float32)
                        f.write(f"{b},{t},cos,{np.mean(v):.6f},{np.percentile(v,25):.6f},{np.percentile(v,75):.6f}\n")
                    if band_to_vals_dl2[b]:
                        v = np.array(band_to_vals_dl2[b], dtype=np.float32)
                        f.write(f"{b},{t},dl2,{np.mean(v):.6f},{np.percentile(v,25):.6f},{np.percentile(v,75):.6f}\n")
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

    if dump_features_dir and feature_blocks:
        os.makedirs(dump_features_dir, exist_ok=True)
        features = np.concatenate(feature_blocks, axis=0)
        labels = np.concatenate(feature_labels, axis=0)
        meta = np.concatenate(feature_meta, axis=0)
        out_path = os.path.join(dump_features_dir, f"teacher_features_{timestamp}.npz")
        np.savez(
            out_path,
            features=features,
            labels=labels,
            meta=meta,
            feature_names=np.array(FEATURE_NAMES),
            label_thresholds=np.array([cos_tau, dl2_rho, kl_max], dtype=np.float32),
        )
        return out_path

    return None


def run_rule_gate(
    engine: BaseEngine,
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

    watchdog_min_cos = cfg.watchdog_min_cos
    watchdog_max_kl = cfg.watchdog_max_kl

    per_seq_latency_ms: List[float] = []
    final_consistency: List[float] = []
    total_start = time.time()

    for prompt in prompts:
        state = engine.encode_prompt(prompt)
        prev_hidden: Optional[List[np.ndarray]] = None
        prev_logits: Optional[np.ndarray] = None

        total_tokens_computed = 0
        total_tokens_possible = 0
        # New: track token×layer×step ops
        num_layers_seen: Optional[int] = None
        baseline_ops = 0
        actual_ops = 0
        skip_stats_rows: List[str] = []

        seq_start = time.time()
        for t in range(num_steps):
            if prev_hidden is None or prev_logits is None:
                out = engine.step(state, t)
                prev_hidden = [h.copy() for h in out.hidden_by_layer]
                prev_logits = out.logits.copy()
                seq_len = out.logits.shape[0]
                if num_layers_seen is None:
                    num_layers_seen = len(out.hidden_by_layer)
                total_tokens_computed += seq_len
                total_tokens_possible += seq_len
                baseline_ops += (num_layers_seen * seq_len)
                actual_ops += (num_layers_seen * seq_len)
                skip_stats_rows.append(f"{t},{seq_len},0,0.0,{num_layers_seen},0")
                continue

            # Get compute mask based on current freeze cooldowns
            compute_mask = gate.get_compute_mask(prev_logits.shape[0])
            mask_to_use: Optional[np.ndarray]
            if compute_mask is None or np.all(compute_mask):
                mask_to_use = None
            else:
                mask_to_use = compute_mask

            prev_hidden_snapshot = [h.copy() for h in prev_hidden]
            prev_logits_snapshot = prev_logits.copy()

            out = engine.step(state, t, compute_mask=mask_to_use)

            cos_now, dl2_now, kl_now, entropy_now, margin_now, step_frac = compute_token_observables(
                out.hidden_by_layer,
                prev_hidden_snapshot,
                out.logits,
                prev_logits_snapshot,
                t,
                num_steps,
            )

            watchdog_trigger = np.any((cos_now < watchdog_min_cos) | (kl_now > watchdog_max_kl))
            if watchdog_trigger:
                mask_to_use = None
                out = engine.step(state, t, compute_mask=None)
                cos_now, dl2_now, kl_now, entropy_now, margin_now, step_frac = compute_token_observables(
                    out.hidden_by_layer,
                    prev_hidden_snapshot,
                    out.logits,
                    prev_logits_snapshot,
                    t,
                    num_steps,
                )

            seq_len = out.logits.shape[0]
            frozen_tokens = 0 if mask_to_use is None else (seq_len - int(np.sum(mask_to_use)))
            frozen_ratio = float(frozen_tokens) / float(seq_len)
            total_tokens_computed += (seq_len - frozen_tokens)
            total_tokens_possible += seq_len
            if num_layers_seen is None:
                num_layers_seen = len(out.hidden_by_layer)
            baseline_ops += (num_layers_seen * seq_len)
            actual_ops += (num_layers_seen * (seq_len - frozen_tokens))
            skip_stats_rows.append(f"{t},{seq_len},{frozen_tokens},{frozen_ratio:.6f},{num_layers_seen},0")

            gate.update(
                feats_cos=cos_now,
                feats_dl2=dl2_now,
                feats_kl=kl_now,
                entropy=entropy_now,
                margin=margin_now,
                num_tokens=out.logits.shape[0],
                compute_mask=mask_to_use,
            )

            prev_hidden = [h.copy() for h in out.hidden_by_layer]
            prev_logits = out.logits.copy()

        per_seq_latency_ms.append((time.time() - seq_start) * 1000.0)

        # Token savings (old): per-step token compute ratio
        token_savings = 1.0 - (total_tokens_computed / max(1, total_tokens_possible))
        # FLOPs-proxy savings (new): token×layer×step
        flops_savings = 1.0 - (actual_ops / max(1, baseline_ops))
        savings_ratios.append(float(flops_savings))
        aggregates.append({
            "mode": "rule_gate",
            "prompt_len": len(prompt),
            "token_savings": float(token_savings),
            "flops_savings": float(flops_savings),
        })

        # Write skip_stats.csv for this prompt
        stats_csv = os.path.join(out_dirs["runs"], f"skip_stats_{int(time.time())}.csv")
        with open(stats_csv, "w", encoding="utf-8") as f:
            f.write("step,seq_len,frozen_tokens,frozen_ratio,num_layers,skipped_layers\n")
            for row in skip_stats_rows:
                f.write(row + "\n")

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


def run_learned_gate(
    engine: BaseEngine,
    prompts: List[str],
    out_dirs: Dict[str, str],
    num_steps: int,
    gate: LearnedGate,
    watchdog_min_cos: float,
    watchdog_max_kl: float,
    consistency_check: bool = False,
) -> None:
    from src.viz.plots import save_hist

    aggregates: List[Dict] = []
    savings_ratios: List[float] = []
    prob_samples: List[float] = []

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

            compute_mask = gate.get_compute_mask(prev_logits.shape[0])
            mask_to_use: Optional[np.ndarray]
            if compute_mask is None or np.all(compute_mask):
                mask_to_use = None
            else:
                mask_to_use = compute_mask

            prev_hidden_snapshot = [h.copy() for h in prev_hidden]
            prev_logits_snapshot = prev_logits.copy()

            out = engine.step(state, t, compute_mask=mask_to_use)

            cos_now, dl2_now, kl_now, entropy_now, margin_now, step_frac = compute_token_observables(
                out.hidden_by_layer,
                prev_hidden_snapshot,
                out.logits,
                prev_logits_snapshot,
                t,
                num_steps,
            )

            watchdog_trigger = np.any((cos_now < watchdog_min_cos) | (kl_now > watchdog_max_kl))
            if watchdog_trigger:
                mask_to_use = None
                out = engine.step(state, t, compute_mask=None)
                cos_now, dl2_now, kl_now, entropy_now, margin_now, step_frac = compute_token_observables(
                    out.hidden_by_layer,
                    prev_hidden_snapshot,
                    out.logits,
                    prev_logits_snapshot,
                    t,
                    num_steps,
                )

            computed_tokens = out.logits.shape[0] if mask_to_use is None else int(np.sum(mask_to_use))
            total_tokens_computed += computed_tokens
            total_tokens_possible += out.logits.shape[0]

            features = np.stack([cos_now, dl2_now, kl_now, entropy_now, margin_now, step_frac], axis=-1).astype(np.float32)
            gate_result = gate.update(
                features=features,
                num_tokens=out.logits.shape[0],
                compute_mask=mask_to_use,
            )

            if gate_result.probabilities is not None:
                prob_samples.extend(gate_result.probabilities.tolist())

            prev_hidden = [h.copy() for h in out.hidden_by_layer]
            prev_logits = out.logits.copy()

        per_seq_latency_ms.append((time.time() - seq_start) * 1000.0)

        savings = 1.0 - (total_tokens_computed / max(1, total_tokens_possible))
        savings_ratios.append(float(savings))
        aggregates.append({"mode": "learned_gate", "prompt_len": len(prompt), "savings": float(savings)})

        if consistency_check:
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
    save_jsonl(os.path.join(out_dirs["runs"], f"learned_gate_{timestamp}.jsonl"), aggregates)
    if savings_ratios:
        save_hist(np.array(savings_ratios), os.path.join(out_dirs["figures"], f"learned_gate_savings_hist_{timestamp}.png"), title="Token compute savings ratio (learned gate)")
    if prob_samples:
        save_hist(np.array(prob_samples), os.path.join(out_dirs["figures"], f"learned_gate_prob_hist_{timestamp}.png"), title="Learned gate freeze probability")

    total_time_s = max(1e-6, time.time() - total_start)
    gpu = query_gpu_utilization() or {}
    summary = {
        "mode": "learned_gate",
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
    with open(os.path.join(out_dirs["runs"], f"learned_gate_summary_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    default_paths = load_default_paths(cfg.get("paths_config"))

    repo_root = Path(__file__).resolve().parents[1]
    mode = args.mode

    phase = coalesce(args.phase, cfg.get("phase"), "P1")
    engine_name = coalesce(args.engine, cfg.get("engine"), "mock")
    outputs_root = coalesce(
        args.outputs_root,
        args.out_dir,
        cfg.get("outputs_root"),
        cfg.get("out_dir"),
        default_paths.get("outputs_root"),
        str((repo_root / "reports").resolve()),
    )
    data_root = coalesce(args.data_root, cfg.get("data_root"), default_paths.get("data_root"))
    models_root = coalesce(args.models_root, cfg.get("models_root"), default_paths.get("models_root"))
    exp_name = coalesce(args.exp_name, cfg.get("exp_name"))
    num_prompts = int(coalesce(args.num_prompts, cfg.get("num_prompts"), 8))
    max_new_tokens = int(coalesce(args.max_new_tokens, cfg.get("max_new_tokens"), 64))
    num_steps = int(coalesce(args.num_steps, cfg.get("num_steps"), 12))
    task = coalesce(args.task, cfg.get("task"), "synthetic")

    dump_features_raw = coalesce(args.dump_features_dir, cfg.get("dump_features_dir"))
    label_cfg = cfg.get("label_thresholds", {})
    label_thresholds = {
        "cosine_tau": float(coalesce(args.label_cos_tau, label_cfg.get("cosine_tau"), 0.97)),
        "delta_l2_rho": float(coalesce(args.label_dl2_rho, label_cfg.get("delta_l2_rho"), 0.05)),
        "kl_max": float(coalesce(args.label_kl_max, label_cfg.get("kl_max"), 0.01)),
    }

    tau = float(coalesce(args.tau, cfg.get("tau"), 0.95))
    rho = float(coalesce(args.rho, cfg.get("rho"), 0.10))
    consecutive_m = int(coalesce(args.m, cfg.get("m"), 2))
    freeze_K = int(coalesce(args.freeze_K, cfg.get("freeze_K"), 2))
    layer_recompute_M = int(coalesce(args.layer_recompute_M, cfg.get("layer_recompute_M"), 2))
    watchdog_cfg = cfg.get("watchdog", {})
    watchdog_min_cos = float(coalesce(args.watchdog_min_cos, watchdog_cfg.get("min_cos"), 0.90))
    watchdog_max_kl = float(coalesce(args.watchdog_max_kl, watchdog_cfg.get("max_kl"), 0.02))

    outputs_root = str(outputs_root)
    run_dir = ensure_output_dirs(outputs_root, phase=phase, mode=mode, engine=engine_name, exp_name=exp_name)
    init_run_logging(run_dir, f"{mode}-{engine_name}")

    dump_features_dir = None
    if dump_features_raw:
        dump_features_dir = dump_features_raw if os.path.isabs(dump_features_raw) else os.path.join(run_dir, dump_features_raw)

    resolved_manifest = {
        "phase": phase,
        "mode": mode,
        "engine": engine_name,
        "outputs_root": outputs_root,
        "data_root": data_root,
        "models_root": models_root,
        "num_prompts": num_prompts,
        "max_new_tokens": max_new_tokens,
        "num_steps": num_steps,
        "task": task,
        "label_thresholds": label_thresholds,
        "rule_gate": {
            "tau": tau,
            "rho": rho,
            "m": consecutive_m,
            "freeze_K": freeze_K,
            "layer_recompute_M": layer_recompute_M,
            "watchdog_min_cos": watchdog_min_cos,
            "watchdog_max_kl": watchdog_max_kl,
        },
    }
    save_manifest(run_dir, {
        "args": vars(args),
        "config": cfg,
        "resolved": resolved_manifest,
    })

    wandb_run = None
    if wandb is not None:
        try:
            os.environ.setdefault("WANDB_MODE", "online")
            try:
                wandb.require("core")
            except Exception:
                pass
            wandb_run = wandb.init(project=os.environ.get("WANDB_PROJECT", "dcllm-delta"), name=os.path.basename(run_dir), dir=run_dir)
        except Exception as exc:  # pragma: no cover
            print(f"[wandb] init failed: {exc}; continuing without W&B logging.")
    else:
        print("[wandb] Package not installed; skipping W&B logging.")

    if engine_name == "mock":
        engine: BaseEngine = MockDiffusionEngine(
            vocab_size=1000,
            num_layers=4,
            hidden_size=64,
            sequence_length=max_new_tokens,
            num_steps=num_steps,
            rng_seed=42,
        )
    elif engine_name == "d2f":
        if D2FDreamEngine is None:
            raise ImportError("D2F engine requested but dependencies are missing. Install torch/transformers and retry.")
        default_model_path = str(Path(models_root) / "checkpoints" / "Dream-v0-Instruct-7B") if models_root else None
        default_lora_path = str(Path(models_root) / "lora" / "D2F_Dream_Base_7B_Lora") if models_root else None
        model_path = coalesce(args.model_path, cfg.get("model_path"), default_model_path)
        lora_path = coalesce(args.lora_path, cfg.get("lora_path"), default_lora_path)
        use_lora = bool(args.use_d2f_lora or cfg.get("use_d2f_lora", False))
        engine = D2FDreamEngine(
            model_name="d2f-small",
            device="cuda",
            model_path=model_path,
            lora_path=lora_path,
            use_lora=use_lora,
            num_steps=num_steps,
            max_seq_len=max_new_tokens,
        )
        # Pass layer reuse cadence into engine (optional)
        try:
            engine.layer_recompute_M = int(coalesce(args.layer_recompute_M, cfg.get("layer_recompute_M"), 0) or 0)
        except Exception:
            engine.layer_recompute_M = 0
    else:
        raise ValueError(f"Unsupported engine: {engine_name}")

    prompts = load_prompts(num_prompts, task, data_root)
    out_dirs = {"figures": os.path.join(run_dir, "figures"), "runs": os.path.join(run_dir, "runs")}

    feature_path: Optional[str] = None
    with Timer(f"run-{mode}"):
        if mode == "teacher":
            feature_path = run_teacher(
                engine,
                prompts,
                out_dirs,
                num_steps,
                dump_features_dir=dump_features_dir,
                label_thresholds=label_thresholds,
            )
            if args.consistency_full_compute and engine_name == "d2f":
                # Full-compute twice to verify stability; write difference report
                diffs_csv = os.path.join(out_dirs["runs"], f"consistency_diffs_{time.strftime('%Y%m%d_%H%M%S')}.csv")
                with open(diffs_csv, "w", encoding="utf-8") as f:
                    f.write("prompt_idx,step,logits_match,hidden_cos_mean\n")
                    for pi, prompt in enumerate(prompts):
                        s1 = engine.encode_prompt(prompt)
                        s2 = engine.encode_prompt(prompt)
                        prev1_h = prev2_h = None
                        prev1_z = prev2_z = None
                        for t in range(num_steps):
                            o1 = engine.step(s1, t, compute_mask=None)
                            o2 = engine.step(s2, t, compute_mask=None)
                            logits_match = int(np.allclose(o1.logits, o2.logits, atol=1e-6))
                            hidden_cos_mean = 1.0
                            if prev1_h is not None and prev2_h is not None:
                                c1 = cosine_similarity_tokens(o1.hidden_by_layer, prev1_h)
                                c2 = cosine_similarity_tokens(o2.hidden_by_layer, prev2_h)
                                hidden_cos_mean = float(0.5 * (np.mean(c1) + np.mean(c2)))
                            f.write(f"{pi},{t},{logits_match},{hidden_cos_mean:.6f}\n")
                            prev1_h = [h.copy() for h in o1.hidden_by_layer]
                            prev2_h = [h.copy() for h in o2.hidden_by_layer]
                            prev1_z = o1.logits.copy()
                            prev2_z = o2.logits.copy()
        elif mode == "rule_gate":
            rule_cfg = RuleGateConfig(
                cosine_tau=tau,
                delta_l2_rho=rho,
                consecutive_m=consecutive_m,
                freeze_K=freeze_K,
                watchdog_min_cos=watchdog_min_cos,
                watchdog_max_kl=watchdog_max_kl,
                layer_recompute_M=layer_recompute_M,
            )
            run_rule_gate(
                engine,
                prompts,
                out_dirs,
                num_steps,
                rule_cfg,
                consistency_check=args.consistency_check,
            )
        elif mode == "learned_gate":
            default_gate_path = str(Path(models_root) / "gates" / "learned_gate_latest.npz") if models_root else None
            weights_path = coalesce(args.learned_gate_weights, cfg.get("learned_gate_weights"), default_gate_path)
            if not weights_path or not os.path.exists(weights_path):
                raise FileNotFoundError(f"Learned gate weights not found at {weights_path}")
            weights = LearnedGateWeights.from_npz(weights_path)
            threshold_override = coalesce(args.learned_gate_threshold, cfg.get("learned_gate_threshold"))
            if threshold_override is not None:
                weights.threshold = float(threshold_override)
            learned_gate_cfg = LearnedGateConfig(
                freeze_K=int(coalesce(args.learned_gate_freeze_K, cfg.get("learned_gate_freeze_K"), freeze_K)),
                min_consecutive=int(coalesce(args.learned_gate_min_consecutive, cfg.get("learned_gate_min_consecutive"), consecutive_m)),
            )
            gate = LearnedGate(weights, learned_gate_cfg)
            run_learned_gate(
                engine,
                prompts,
                out_dirs,
                num_steps,
                gate,
                watchdog_min_cos=watchdog_min_cos,
                watchdog_max_kl=watchdog_max_kl,
                consistency_check=args.consistency_check,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    if feature_path:
        print(f"[teacher] token features saved to {feature_path}")

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
