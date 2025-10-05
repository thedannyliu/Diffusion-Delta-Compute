from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Optional

import numpy as np


@dataclass
class ConformalRiskCalibrator:
    """Maintains an online conformal threshold for risk scores.

    The implementation follows a sliding-window quantile estimator sized to
    achieve a user-specified false-positive upper bound ``delta``. Unsafe tokens
    (identified via watchdog triggers, quality regressions, or oracle labels)
    should be registered via :meth:`register`.
    """

    delta: float = 0.01
    window_size: int = 4096
    min_size: int = 128
    initial_quantile: float = 0.20

    def __post_init__(self) -> None:
        self._unsafe_scores: Deque[float] = deque(maxlen=self.window_size)
        # ``initial_quantile`` is expressed as a probability in [0, 1]. Store a
        # clamped copy so we can safely reuse it for fallback bootstrapping.
        self._initial_quantile = float(np.clip(self.initial_quantile, 0.0, 1.0))
        # ``_quantile`` tracks the calibrated risk value. ``inf`` denotes that no
        # samples have been registered yet, allowing callers to fall back to a
        # bootstrap estimate derived from the current batch of scores.
        self._quantile: float = float("inf")

    def register(self, scores: np.ndarray, unsafe_mask: Optional[np.ndarray] = None) -> None:
        """Register a batch of scores.

        Parameters
        ----------
        scores:
            Risk scores for the current batch of tokens.
        unsafe_mask:
            Optional boolean mask (same shape as ``scores``) indicating which
            tokens were discovered to be unsafe. When ``None`` the full batch is
            treated as unsafe, which is conservative.
        """

        if scores.size == 0:
            return
        if unsafe_mask is None:
            selected = scores.reshape(-1)
        else:
            if unsafe_mask.shape != scores.shape:
                raise ValueError("unsafe_mask must match scores shape")
            selected = scores[unsafe_mask]
        if selected.size == 0:
            return
        for value in selected.astype(np.float64).tolist():
            self._unsafe_scores.append(float(value))
        self._recompute_quantile()

    def _recompute_quantile(self) -> None:
        count = len(self._unsafe_scores)
        if count == 0:
            self._quantile = float("inf")
            return

        scores = np.fromiter(self._unsafe_scores, dtype=np.float64)
        target_prob = float(1.0 - self.delta)
        if count < self.min_size:
            target_prob = max(target_prob, self._initial_quantile)
        target_prob = float(np.clip(target_prob, 0.0, 1.0))

        if target_prob >= 1.0:
            self._quantile = float(np.max(scores))
        else:
            # ``method='higher'`` mirrors the +1 conformal correction.
            self._quantile = float(np.quantile(scores, target_prob, method="higher"))

    @property
    def threshold(self) -> float:
        return float(self._quantile)

    @property
    def is_ready(self) -> bool:
        return len(self._unsafe_scores) >= self.min_size and not np.isinf(self._quantile)

    def effective_threshold(self, candidate_scores: Optional[np.ndarray] = None) -> float:
        """Return the calibrated threshold, bootstrapping if needed.

        When no unsafe samples have been registered yet we fall back to a
        quantile computed from ``candidate_scores`` (typically the current
        batch). This keeps the downstream gates from becoming overly
        conservative in early steps while still honouring the configured
        ``initial_quantile``.
        """

        if not np.isinf(self._quantile):
            return float(self._quantile)
        if candidate_scores is None or candidate_scores.size == 0:
            return float(self._quantile)
        prob = self._initial_quantile if self._initial_quantile > 0.0 else 0.5
        prob = float(np.clip(prob, 0.0, 1.0))
        if prob >= 1.0:
            return float(np.max(candidate_scores))
        return float(np.quantile(candidate_scores.astype(np.float64), prob, method="higher"))

    def reset(self) -> None:
        self._unsafe_scores.clear()
        self._quantile = float("inf")


def compute_risk_score(
    cosine: np.ndarray,
    delta_l2: np.ndarray,
    kl: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Compute monotone risk score per token.

    The score follows ``max{ΔL2, KL, 1 - cos}`` which empirically correlates
    with unsafe skipping events. All operands are assumed to already be
    normalised (e.g., ΔL2 divided by σ_t).
    """

    if not (cosine.shape == delta_l2.shape == kl.shape):
        raise ValueError("cosine, delta_l2, kl must share the same shape")
    risk_components = np.stack([
        delta_l2,
        kl,
        1.0 - cosine,
    ], axis=0)
    return np.max(risk_components, axis=0) + float(epsilon)


@dataclass
class StepwiseConformalCalibrator:
    """Per-step Mondrian conformal calibrator with optional smoothing and clipping."""

    delta: float = 0.01
    num_steps: Optional[int] = None
    initial_quantile: float = 0.60
    ema_alpha: Optional[float] = 0.9
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None
    low_support: int = 50
    window_size: Optional[int] = None
    init_thresholds: Optional[Dict[int, float]] = None

    def __post_init__(self) -> None:
        maxlen = self.window_size if self.window_size and self.window_size > 0 else None
        self._unsafe_scores: Dict[int, Deque[float]] = defaultdict(lambda: deque(maxlen=maxlen))
        self._thresholds: Dict[int, float] = {}
        if self.init_thresholds:
            for k, v in self.init_thresholds.items():
                self._thresholds[int(k)] = float(v)
        self._raw_thresholds: Dict[int, float] = {}

    def register(self, step: int, scores: np.ndarray, unsafe_mask: Optional[np.ndarray] = None) -> None:
        if scores.size == 0:
            return
        if unsafe_mask is None:
            selected = scores.reshape(-1)
        else:
            if unsafe_mask.shape != scores.shape:
                raise ValueError("unsafe_mask must match scores shape")
            selected = scores[unsafe_mask]
        if selected.size == 0:
            return
        bucket = self._unsafe_scores[int(step)]
        for value in selected.astype(np.float64).tolist():
            bucket.append(float(value))
        self._update_threshold(int(step))

    def _compute_quantile(self, step: int) -> Optional[float]:
        bucket = self._unsafe_scores.get(step)
        if not bucket:
            return None
        arr = np.fromiter(bucket, dtype=np.float64)
        n = arr.size
        if n < max(1, self.low_support):
            return None
        k = int(np.ceil((1.0 - self.delta) * (n + 1)))
        k = max(1, min(k, n))
        arr.sort()
        return float(arr[k - 1])

    def _apply_smoothing(self, step: int, value: float) -> float:
        if self.ema_alpha is None:
            return value
        prev = self._raw_thresholds.get(step)
        if prev is None:
            return value
        return float(self.ema_alpha * prev + (1.0 - self.ema_alpha) * value)

    def _apply_clip(self, value: float) -> float:
        lower = self.clip_min if self.clip_min is not None else -np.inf
        upper = self.clip_max if self.clip_max is not None else np.inf
        return float(np.clip(value, lower, upper))

    def _update_threshold(self, step: int) -> None:
        quantile = self._compute_quantile(step)
        if quantile is None:
            return
        smoothed = self._apply_smoothing(step, quantile)
        clipped = self._apply_clip(smoothed)
        self._raw_thresholds[step] = clipped
        self._thresholds[step] = clipped

    def effective_threshold(self, step: int, candidate_scores: Optional[np.ndarray] = None) -> float:
        step_key = int(step)
        if step_key in self._thresholds:
            return float(self._thresholds[step_key])
        self._update_threshold(step_key)
        if step_key in self._thresholds:
            return float(self._thresholds[step_key])
        if candidate_scores is not None and candidate_scores.size > 0 and self.initial_quantile is not None:
            prob = float(np.clip(self.initial_quantile, 0.0, 1.0))
            fallback = float(np.quantile(candidate_scores.astype(np.float64), prob, method="higher"))
            fallback = self._apply_clip(fallback)
            self._thresholds[step_key] = fallback
            return fallback
        return float("inf")

    def thresholds(self) -> Dict[int, float]:
        return dict(self._thresholds)
