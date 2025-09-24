from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .rule_gate import StepGateResult


@dataclass
class LearnedGateWeights:
    weights: np.ndarray  # shape [F]
    bias: float
    mean: np.ndarray  # shape [F]
    std: np.ndarray  # shape [F]
    threshold: float

    @classmethod
    def from_npz(cls, path: str) -> "LearnedGateWeights":
        data = np.load(path)
        weights = data["weights"].astype(np.float32)
        bias = float(data["bias"].item())
        mean = data["mean"].astype(np.float32)
        std = data["std"].astype(np.float32)
        threshold = float(data.get("threshold", np.array([0.5], dtype=np.float32)).item())
        return cls(weights=weights, bias=bias, mean=mean, std=std, threshold=threshold)


@dataclass
class LearnedGateConfig:
    freeze_K: int = 2
    min_consecutive: int = 1


class LearnedGate:
    def __init__(self, weights: LearnedGateWeights, cfg: LearnedGateConfig) -> None:
        self.weights = weights
        self.cfg = cfg
        self._counters: Optional[np.ndarray] = None
        self._cooldown: Optional[np.ndarray] = None

    def _ensure_state(self, num_tokens: int) -> None:
        if self._counters is None or self._counters.shape[0] != num_tokens:
            self._counters = np.zeros((num_tokens,), dtype=np.int32)
            self._cooldown = np.zeros((num_tokens,), dtype=np.int32)

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        std = np.where(self.weights.std == 0.0, 1.0, self.weights.std)
        return (features - self.weights.mean) / std

    def _predict_probabilities(self, features: np.ndarray) -> np.ndarray:
        normalized = self._normalize(features)
        logits = normalized @ self.weights.weights + self.weights.bias
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs.astype(np.float32)

    def get_compute_mask(self, num_tokens: int) -> np.ndarray:
        self._ensure_state(num_tokens)
        assert self._cooldown is not None
        return (self._cooldown <= 0)

    def update(
        self,
        features: np.ndarray,
        num_tokens: int,
        compute_mask: Optional[np.ndarray] = None,
    ) -> StepGateResult:
        self._ensure_state(num_tokens)
        counters = self._counters
        cooldown = self._cooldown

        probs = self._predict_probabilities(features)
        above_threshold = probs >= self.weights.threshold

        if compute_mask is None:
            active = np.ones_like(above_threshold, dtype=bool)
        else:
            active = compute_mask.astype(bool)

        not_frozen = cooldown <= 0
        update_mask = active & not_frozen

        counters[update_mask & above_threshold] += 1
        counters[update_mask & (~above_threshold)] = 0

        start_freeze = (
            (counters >= self.cfg.min_consecutive)
            & (cooldown <= 0)
            & active
            & above_threshold
        )
        cooldown[start_freeze] = self.cfg.freeze_K

        freezing_now = cooldown > 0
        cooldown[freezing_now] -= 1
        cooldown[cooldown < 0] = 0

        compute_next = (cooldown <= 0)
        self._counters = counters
        self._cooldown = cooldown
        return StepGateResult(
            compute_mask=compute_next,
            counters=counters.copy(),
            probabilities=probs,
        )

    def reset(self) -> None:
        self._counters = None
        self._cooldown = None
