from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np


DEFAULT_QUANTILES: Sequence[float] = (0.10, 0.25, 0.50, 0.75, 0.88, 0.90, 0.95, 0.99)


def _format_quantile_key(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def compute_step_metric_quantiles(
    samples: Mapping[int, List[np.ndarray]],
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> Dict[int, Dict[str, float]]:
    result: Dict[int, Dict[str, float]] = {}
    for step, blocks in samples.items():
        if not blocks:
            continue
        concatenated = np.concatenate([np.asarray(block, dtype=np.float32).reshape(-1) for block in blocks if np.size(block) > 0])
        if concatenated.size == 0:
            continue
        quantile_map: Dict[str, float] = {}
        for q in quantiles:
            key = _format_quantile_key(float(q))
            quantile_map[key] = float(np.quantile(concatenated, float(q), method="higher"))
        result[int(step)] = quantile_map
    return result


def _safe_float(value: float) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    raise TypeError(f"Expected float-compatible value, received {type(value)!r}")


def dump_step_quantiles(
    path: str,
    metrics: Mapping[str, Dict[int, Dict[str, float]]],
    *,
    metadata: Optional[MutableMapping[str, object]] = None,
) -> None:
    payload: Dict[str, object] = {
        "metadata": dict(metadata or {}),
        "steps": {},
    }
    step_indices = sorted({step for metric_map in metrics.values() for step in metric_map.keys()})
    for step in step_indices:
        step_entry: Dict[str, Dict[str, float]] = {}
        for metric_name, metric_map in metrics.items():
            if step not in metric_map:
                continue
            step_entry[metric_name] = metric_map[step]
        payload["steps"][str(step)] = step_entry
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)


@dataclass
class StepQuantileTable:
    steps: Dict[str, Dict[str, Dict[str, float]]]
    quantiles: Sequence[float]
    metadata: Dict[str, object]

    @classmethod
    def load(cls, path: str) -> "StepQuantileTable":
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        steps = data.get("steps", {})
        meta = data.get("metadata", {})
        quantiles = meta.get("quantiles")
        if quantiles is None:
            # Infer from first step if absent.
            quantile_keys: Iterable[str] = []
            for entry in steps.values():
                if entry:
                    quantile_keys = next(iter(entry.values())).keys()
                    break
            quantiles = []
            for key in quantile_keys:
                try:
                    quantiles.append(float(key))
                except Exception:
                    continue
        return cls(steps=steps, quantiles=tuple(float(q) for q in quantiles), metadata=meta)

    def value(self, metric: str, step: int, quantile: float) -> float:
        step_key = str(step)
        if step_key not in self.steps:
            raise KeyError(f"Step {step} not present in quantile table")
        metric_entry = self.steps[step_key]
        if metric not in metric_entry:
            raise KeyError(f"Metric '{metric}' missing for step {step}")
        key = _format_quantile_key(float(quantile))
        # Exact match first
        if key in metric_entry[metric]:
            return _safe_float(metric_entry[metric][key])
        # Fallback: choose the nearest available quantile key by numeric distance.
        # This makes the consumer robust to requesting 0.92 when only {0.90, 0.95} exist, etc.
        candidates = []
        for k in metric_entry[metric].keys():
            try:
                candidates.append((abs(float(k) - float(quantile)), k))
            except Exception:
                continue
        if candidates:
            _, nearest_key = min(candidates, key=lambda x: x[0])
            return _safe_float(metric_entry[metric][nearest_key])
        available = ", ".join(sorted(metric_entry[metric].keys()))
        raise KeyError(f"Quantile {quantile} missing for metric '{metric}' at step {step}; available: {available}")

    @property
    def num_steps(self) -> Optional[int]:
        return int(self.metadata["num_steps"]) if "num_steps" in self.metadata else None


def apply_ema(values: Sequence[float], alpha: float) -> List[float]:
    if not values:
        return []
    if not (0.0 < alpha < 1.0):
        return list(values)
    smoothed: List[float] = []
    acc: Optional[float] = None
    for val in values:
        fv = float(val)
        if acc is None:
            acc = fv
        else:
            acc = alpha * acc + (1.0 - alpha) * fv
        smoothed.append(acc)
    return smoothed


def clip_sequence(values: Sequence[float], clip_min: Optional[float], clip_max: Optional[float]) -> List[float]:
    if clip_min is None and clip_max is None:
        return list(float(v) for v in values)
    return [float(np.clip(v, clip_min if clip_min is not None else -math.inf, clip_max if clip_max is not None else math.inf)) for v in values]
