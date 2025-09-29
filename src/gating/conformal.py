from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

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
        self._quantile: float = float(self.initial_quantile)

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
        if len(self._unsafe_scores) < self.min_size:
            return
        # Conformal quantile (empirical upper quantile with +1 correction)
        scores = np.fromiter(self._unsafe_scores, dtype=np.float64)
        n = scores.size
        rank = int(np.ceil((n + 1) * (1.0 - self.delta)))
        rank = np.clip(rank, 1, n)
        sorted_scores = np.sort(scores)
        self._quantile = float(sorted_scores[rank - 1])

    @property
    def threshold(self) -> float:
        return float(self._quantile)

    def reset(self) -> None:
        self._unsafe_scores.clear()
        self._quantile = float(self.initial_quantile)


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

