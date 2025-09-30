from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .rule_gate import StepGateResult


@dataclass
class LearnedGateWeights:
    weights: np.ndarray  # shape [F]
    bias: float
    mean: np.ndarray  # shape [F]
    std: np.ndarray  # shape [F]
    threshold: float
    feature_names: Optional[List[str]] = None

    @classmethod
    def from_npz(cls, path: str) -> "LearnedGateWeights":
        data = np.load(path)
        weights = data["weights"].astype(np.float32)
        bias = float(data["bias"].item())
        mean = data["mean"].astype(np.float32)
        std = data["std"].astype(np.float32)
        threshold = float(data.get("threshold", np.array([0.5], dtype=np.float32)).item())
        feature_names = None
        if "feature_names" in data:
            feature_names = [str(x) for x in data["feature_names"].tolist()]
        return cls(weights=weights, bias=bias, mean=mean, std=std, threshold=threshold, feature_names=feature_names)


@dataclass
class LearnedGateConfig:
    freeze_K: int = 2
    min_consecutive: int = 1
    gamma_margin: float = 0.08


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
        dim = min(features.shape[-1], self.weights.mean.shape[0])
        if features.shape[-1] != dim:
            features = features[..., :dim]
        return (features[..., :dim] - self.weights.mean[:dim]) / std[:dim]

    def _predict_probabilities(self, features: np.ndarray) -> np.ndarray:
        normalized = self._normalize(features)
        # Align feature and weight dimensions defensively
        weight_vec = self.weights.weights
        dim = min(normalized.shape[-1], weight_vec.shape[0])
        logits = normalized[..., :dim] @ weight_vec[:dim] + self.weights.bias
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
        risk_threshold: float,
        feature_margin: Optional[np.ndarray] = None,
        compute_mask: Optional[np.ndarray] = None,
        budget_controller: Optional[object] = None,
    ) -> StepGateResult:
        self._ensure_state(num_tokens)
        counters = self._counters
        cooldown = self._cooldown

        probs = self._predict_probabilities(features)
        above_threshold = probs >= self.weights.threshold
        risk_scores = 1.0 - probs

        if feature_margin is None:
            if features.shape[-1] >= 5:
                feature_margin = features[:, 4]
            else:
                feature_margin = np.ones_like(probs)
        margin_ok = feature_margin >= self.cfg.gamma_margin
        safe = above_threshold & margin_ok & (risk_scores <= risk_threshold)

        if compute_mask is None:
            active = np.ones_like(above_threshold, dtype=bool)
        else:
            active = compute_mask.astype(bool)

        not_frozen = cooldown <= 0
        update_mask = active & not_frozen

        counters[update_mask & safe] += 1
        counters[update_mask & (~safe)] = 0

        start_freeze = (
            (counters >= self.cfg.min_consecutive)
            & (cooldown <= 0)
            & active
            & safe
        )
        cooldown[start_freeze] = self.cfg.freeze_K

        freezing_now = cooldown > 0
        cooldown[freezing_now] -= 1
        cooldown[cooldown < 0] = 0

        compute_next = (cooldown <= 0)
        if budget_controller is not None:
            from .budget import BudgetController  # local import to avoid cycle

            if not isinstance(budget_controller, BudgetController):
                raise TypeError("budget_controller must be a BudgetController")
            adjusted = budget_controller.select(compute_next.astype(bool), risk_scores)
            newly_frozen = (~adjusted) & compute_next
            cooldown[newly_frozen] = np.maximum(cooldown[newly_frozen], 1)
            compute_next = adjusted
        self._counters = counters
        self._cooldown = cooldown
        return StepGateResult(
            compute_mask=compute_next,
            counters=counters.copy(),
            probabilities=probs,
            risk_scores=risk_scores,
            freeze_mask=~compute_next.astype(bool),
        )

    def reset(self) -> None:
        self._counters = None
        self._cooldown = None


