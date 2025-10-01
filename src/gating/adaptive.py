from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AdaptiveSchedulerConfig:
    lte_eps: float = 0.2
    min_consecutive: int = 2
    max_stride: int = 4
    base_stride: int = 1
    use_normalized_lte: bool = False


class AdaptiveScheduler:
    """Lightweight stride controller based on LTE estimates."""

    def __init__(self, cfg: AdaptiveSchedulerConfig) -> None:
        self.cfg = cfg
        self._current_stride: int = cfg.base_stride
        self._streak: int = 0

    @property
    def stride(self) -> int:
        return self._current_stride

    def observe(self, lte_value: float, lte_threshold: float, risk_value: float, risk_threshold: float) -> None:
        """Update internal stride state based on observed LTE and risk.

        ``risk_value`` is compared against the supplied ``risk_threshold`` so
        that callers can adapt the decision boundary based on conformal
        calibration.
        """

        if (lte_value <= lte_threshold) and (risk_value <= risk_threshold):
            self._streak += 1
            if self._streak >= self.cfg.min_consecutive:
                self._current_stride = min(self.cfg.max_stride, self._current_stride + 1)
                self._streak = 0
        else:
            self._current_stride = self.cfg.base_stride
            self._streak = 0

    def reset(self) -> None:
        self._current_stride = self.cfg.base_stride
        self._streak = 0


def heun_lte(estimate_a: np.ndarray, estimate_b: np.ndarray, prev: np.ndarray, eps: float = 1e-6) -> float:
    """Compute LTE between Euler (a) and Heun (b) hidden states."""

    if estimate_a.shape != estimate_b.shape:
        raise ValueError("estimate_a and estimate_b must have the same shape")
    diff = np.linalg.norm(estimate_b - estimate_a)
    denom = np.linalg.norm(prev) + eps
    return float(diff / denom)
