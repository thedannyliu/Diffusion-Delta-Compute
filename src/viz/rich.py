from __future__ import annotations

import os
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .plots import _try_import_matplotlib, save_heatmap, save_hist


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _stack_series(series: Sequence[Sequence[float]]) -> np.ndarray:
    if not series:
        return np.empty((0, 0), dtype=np.float32)
    max_len = max(len(seq) for seq in series)
    if max_len == 0:
        return np.empty((0, len(series)), dtype=np.float32)
    arr = np.full((max_len, len(series)), np.nan, dtype=np.float32)
    for col, seq in enumerate(series):
        if not seq:
            continue
        arr[: len(seq), col] = np.asarray(seq, dtype=np.float32)
    return arr


def _save_violin(values: Sequence[Sequence[float]], labels: Sequence[str], path: str, title: str) -> None:
    _ensure_dir(os.path.dirname(path))
    plt = _try_import_matplotlib()
    filtered = [(np.asarray(v, dtype=np.float32), lbl) for v, lbl in zip(values, labels) if len(v) > 0]
    if plt is None:
        flat = np.concatenate([v for v, _ in filtered]) if filtered else np.array([], dtype=np.float32)
        if flat.size > 0:
            np.savetxt(path.replace(".png", ".csv"), flat.reshape(-1, 1), delimiter=",", header="value", comments="")
        return
    plt.figure(figsize=(6, 4))
    if not filtered:
        plt.close()
        return
    data = [v for v, _ in filtered]
    plt.violinplot(data, showmeans=True, showextrema=False)
    plt.xticks(np.arange(1, len(data) + 1), [lbl for _, lbl in filtered])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_line(xs: Sequence[float], ys: Sequence[float], path: str, title: str, xlabel: str, ylabel: str) -> None:
    _ensure_dir(os.path.dirname(path))
    plt = _try_import_matplotlib()
    if plt is None:
        data = np.stack([np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)], axis=-1)
        np.savetxt(path.replace(".png", ".csv"), data, delimiter=",", header="x,y", comments="")
        return
    plt.figure(figsize=(6, 3.5))
    plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_density(values: Sequence[float], path: str, title: str, xlabel: str) -> None:
    _ensure_dir(os.path.dirname(path))
    plt = _try_import_matplotlib()
    arr = np.asarray(values, dtype=np.float32)
    if plt is None or arr.size == 0:
        if arr.size > 0:
            np.savetxt(path.replace(".png", ".csv"), arr.reshape(-1, 1), delimiter=",", header="value", comments="")
        return
    plt.figure(figsize=(6, 3.5))
    plt.hist(arr, bins=min(30, max(5, int(np.sqrt(arr.size)))), density=True, color="teal", alpha=0.8)
    p50 = np.percentile(arr, 50)
    p90 = np.percentile(arr, 90)
    plt.axvline(p50, color="black", linestyle="--", label="p50")
    plt.axvline(p90, color="red", linestyle=":", label="p90")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def render_teacher_suite(
    figures_dir: str,
    timestamp: str,
    cos_means: Sequence[Sequence[float]],
    dl2_means: Sequence[Sequence[float]],
    kl_means: Sequence[Sequence[float]],
    ent_means: Sequence[Sequence[float]],
    dl2_norm_means: Sequence[Sequence[float]],
    cos_hist: Sequence[float],
    dl2_hist: Sequence[float],
    dl2_norm_hist: Sequence[float],
    stability_lengths: Sequence[int],
    latencies_ms: Sequence[float],
    mse_series: Sequence[Sequence[float]],
) -> None:
    _ensure_dir(figures_dir)

    # Additional heatmaps: KL and ΔL2_norm
    kl_arr = _stack_series(kl_means)
    if kl_arr.size:
        save_heatmap(np.nan_to_num(kl_arr).T, os.path.join(figures_dir, f"teacher_kl_heatmap_{timestamp}.png"), title="KL mean per step")
    dl2_norm_arr = _stack_series(dl2_norm_means)
    if dl2_norm_arr.size:
        save_heatmap(np.nan_to_num(dl2_norm_arr).T, os.path.join(figures_dir, f"teacher_dl2norm_heatmap_{timestamp}.png"), title="ΔL2 norm mean per step")

    # Histograms + violin plots for cosine / ΔL2 metrics
    if cos_hist:
        _save_violin([cos_hist], ["cos"], os.path.join(figures_dir, f"teacher_cosine_violin_{timestamp}.png"), "Cosine distribution")
    if dl2_hist:
        _save_violin([dl2_hist], ["ΔL2"], os.path.join(figures_dir, f"teacher_dl2_violin_{timestamp}.png"), "ΔL2 distribution")
    if dl2_norm_hist:
        _save_violin([dl2_norm_hist], ["ΔL2_norm"], os.path.join(figures_dir, f"teacher_dl2norm_violin_{timestamp}.png"), "ΔL2/σ distribution")

    # Token stability lengths
    if stability_lengths:
        save_hist(np.array(stability_lengths, dtype=np.float32), os.path.join(figures_dir, f"teacher_stability_lengths_{timestamp}.png"), title="Token stability span (steps)")

    # Latency tails density
    if latencies_ms:
        _save_density(latencies_ms, os.path.join(figures_dir, f"teacher_latency_tails_{timestamp}.png"), "Latency distribution", "latency (ms)")

    # Average MSE per step across prompts
    mse_arr = _stack_series(mse_series)
    if mse_arr.size:
        mean_curve = np.nanmean(mse_arr, axis=1)
        steps = np.arange(1, mean_curve.shape[0] + 1, dtype=np.int32)
        _save_line(steps, mean_curve, os.path.join(figures_dir, f"teacher_mse_curve_{timestamp}.png"), "Mean per-step MSE", "step", "MSE")

    # Entropy curve for reference
    ent_arr = _stack_series(ent_means)
    if ent_arr.size:
        mean_entropy = np.nanmean(ent_arr, axis=1)
        steps = np.arange(mean_entropy.shape[0], dtype=np.int32)
        _save_line(steps, mean_entropy, os.path.join(figures_dir, f"teacher_entropy_curve_{timestamp}.png"), "Mean entropy", "step", "entropy")


def _aggregate_by_step(values: Dict[int, Sequence[float]]) -> np.ndarray:
    if not values:
        return np.empty((0,), dtype=np.float32)
    max_step = max(values)
    arr = np.full((max_step + 1,), np.nan, dtype=np.float32)
    for step, seq in values.items():
        if not seq:
            continue
        arr[step] = float(np.mean(seq))
    return arr


def render_rule_gate_suite(
    figures_dir: str,
    timestamp: str,
    savings_ratios: Sequence[float],
    freeze_ratios_by_step: Dict[int, Sequence[float]],
    risk_threshold_history: Sequence[float],
    delta_violation_history: Sequence[float],
    latencies_ms: Sequence[float],
) -> None:
    _ensure_dir(figures_dir)

    freeze_curve = _aggregate_by_step(freeze_ratios_by_step)
    if freeze_curve.size:
        steps = np.arange(freeze_curve.shape[0])
        _save_line(steps, np.nan_to_num(freeze_curve), os.path.join(figures_dir, f"rule_gate_freeze_curve_{timestamp}.png"), "Mean frozen ratio per step", "step", "freeze fraction")

    if risk_threshold_history:
        xs = np.arange(len(risk_threshold_history))
        _save_line(xs, risk_threshold_history, os.path.join(figures_dir, f"rule_gate_risk_threshold_{timestamp}.png"), "Risk threshold trace", "update", "threshold")

    if delta_violation_history:
        xs = np.arange(len(delta_violation_history))
        _save_line(xs, delta_violation_history, os.path.join(figures_dir, f"rule_gate_delta_violation_{timestamp}.png"), "Delta violation rate", "update", "fraction")

    if latencies_ms:
        _save_density(latencies_ms, os.path.join(figures_dir, f"rule_gate_latency_tails_{timestamp}.png"), "Latency distribution", "latency (ms)")

    if savings_ratios:
        save_hist(np.array(savings_ratios, dtype=np.float32), os.path.join(figures_dir, f"rule_gate_savings_violin_{timestamp}.png"), title="FLOPs savings distribution")


def render_learned_gate_suite(
    figures_dir: str,
    timestamp: str,
    savings_ratios: Sequence[float],
    freeze_ratios_by_step: Dict[int, Sequence[float]],
    risk_threshold_history: Sequence[float],
    delta_violation_history: Sequence[float],
    latencies_ms: Sequence[float],
    prob_samples: Sequence[float],
) -> None:
    render_rule_gate_suite(
        figures_dir,
        timestamp,
        savings_ratios,
        freeze_ratios_by_step,
        risk_threshold_history,
        delta_violation_history,
        latencies_ms,
    )

    if prob_samples:
        save_hist(np.array(prob_samples, dtype=np.float32), os.path.join(figures_dir, f"learned_gate_prob_density_{timestamp}.png"), title="Learned gate freeze probability")


def render_adaptive_suite(
    figures_dir: str,
    timestamp: str,
    skip_estimates: Sequence[float],
    lte_traces: Sequence[Sequence[float]],
    risk_traces: Sequence[Sequence[float]],
    stride_traces: Sequence[Sequence[int]],
    lte_threshold_traces: Sequence[Sequence[float]],
    cfg,
) -> None:
    _ensure_dir(figures_dir)

    lte_arr = _stack_series(lte_traces)
    if lte_arr.size:
        mean_lte = np.nanmean(lte_arr, axis=1)
        steps = np.arange(1, mean_lte.shape[0] + 1)
        _save_line(steps, mean_lte, os.path.join(figures_dir, f"adaptive_lte_curve_{timestamp}.png"), "Mean LTE", "effective step", "LTE value")

    risk_arr = _stack_series(risk_traces)
    if risk_arr.size:
        mean_risk = np.nanmean(risk_arr, axis=1)
        steps = np.arange(1, mean_risk.shape[0] + 1)
        _save_line(steps, mean_risk, os.path.join(figures_dir, f"adaptive_risk_curve_{timestamp}.png"), "Mean risk (p90)", "effective step", "risk")

    stride_arr = _stack_series(stride_traces)
    if stride_arr.size:
        mean_stride = np.nanmean(stride_arr, axis=1)
        steps = np.arange(mean_stride.shape[0])
        _save_line(steps, mean_stride, os.path.join(figures_dir, f"adaptive_stride_curve_{timestamp}.png"), "Scheduler stride", "step", "stride")

    threshold_arr = _stack_series(lte_threshold_traces)
    if threshold_arr.size:
        mean_thresh = np.nanmean(threshold_arr, axis=1)
        steps = np.arange(1, mean_thresh.shape[0] + 1)
        _save_line(steps, mean_thresh, os.path.join(figures_dir, f"adaptive_lte_threshold_{timestamp}.png"), "LTE threshold (effective)", "effective step", "threshold")

    if skip_estimates:
        _save_density(skip_estimates, os.path.join(figures_dir, f"adaptive_skip_ratio_density_{timestamp}.png"), "Estimated skip ratio", "skip ratio")

    # Document config for reproducibility
    cfg_path = os.path.join(figures_dir, f"adaptive_config_{timestamp}.txt")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(str(cfg))
