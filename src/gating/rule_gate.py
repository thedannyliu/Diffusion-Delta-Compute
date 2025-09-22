from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RuleGateConfig:
    cosine_tau: float = 0.95
    delta_l2_rho: float = 0.10
    consecutive_m: int = 2
    freeze_K: int = 2
    watchdog_min_cos: float = 0.90
    watchdog_max_kl: float = 0.02
    layer_recompute_M: int = 2


@dataclass
class StepGateResult:
    compute_mask: Optional[np.ndarray]
    counters: np.ndarray


class RuleGate:
    """Stateful rule-based per-token freeze gate.

    Maintains per-token stability counters and freeze cooldowns.
    """

    def __init__(self, cfg: RuleGateConfig) -> None:
        self.cfg = cfg
        self._counters: Optional[np.ndarray] = None
        self._cooldown: Optional[np.ndarray] = None  # remaining frozen steps per token

    def _ensure_state(self, num_tokens: int) -> None:
        if self._counters is None or self._counters.shape[0] != num_tokens:
            self._counters = np.zeros((num_tokens,), dtype=np.int32)
            self._cooldown = np.zeros((num_tokens,), dtype=np.int32)

    def get_compute_mask(self, num_tokens: int) -> np.ndarray:
        self._ensure_state(num_tokens)
        assert self._cooldown is not None
        return (self._cooldown <= 0)

    def update(
        self,
        feats_cos: np.ndarray,
        feats_dl2: np.ndarray,
        feats_kl: np.ndarray,
        entropy: np.ndarray,
        margin: np.ndarray,
        num_tokens: int,
    ) -> StepGateResult:
        self._ensure_state(num_tokens)
        counters = self._counters
        cooldown = self._cooldown

        stable = (feats_cos >= self.cfg.cosine_tau) | (feats_dl2 <= self.cfg.delta_l2_rho)

        not_frozen = cooldown <= 0
        counters[not_frozen & stable] += 1
        counters[not_frozen & (~stable)] = 0

        start_freeze = (counters >= self.cfg.consecutive_m) & (cooldown <= 0)
        cooldown[start_freeze] = self.cfg.freeze_K

        # Decrement cooldown at end of step
        freezing_now = cooldown > 0
        cooldown[freezing_now] -= 1

        compute_mask = (cooldown <= 0)
        self._counters = counters
        self._cooldown = cooldown
        return StepGateResult(compute_mask=compute_mask, counters=counters.copy())


