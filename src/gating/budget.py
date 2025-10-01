from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BudgetController:
    """Greedy per-step compute budget controller.

    The controller ensures that the number of tokens recomputed at a step does
    not exceed ``budget_fraction`` of the teacher compute. It keeps the highest
    risk tokens and freezes the rest.
    """

    budget_fraction: float = 1.0

    def select(self, compute_mask: np.ndarray, risk_scores: np.ndarray) -> np.ndarray:
        if compute_mask.dtype != bool:
            compute_mask = compute_mask.astype(bool)
        seq_len = compute_mask.shape[0]
        if seq_len == 0:
            return compute_mask
        max_compute = int(np.ceil(self.budget_fraction * seq_len))
        max_compute = np.clip(max_compute, 0, seq_len)
        # Already under budget → no change
        compute_indices = np.nonzero(compute_mask)[0]
        if compute_indices.size <= max_compute:
            return compute_mask

        # Sort currently-computed tokens by decreasing risk (keep the riskiest)
        scores = risk_scores[compute_indices]
        order = np.argsort(-scores)
        keep = compute_indices[order[:max_compute]]

        new_mask = np.zeros_like(compute_mask, dtype=bool)
        new_mask[keep] = True
        return new_mask

    def remaining_fraction(self, compute_mask: np.ndarray) -> float:
        if compute_mask.size == 0:
            return 0.0
        computed = float(np.count_nonzero(compute_mask))
        return 1.0 - (computed / float(compute_mask.size))


def apply_budget(
    compute_mask: Optional[np.ndarray],
    risk_scores: np.ndarray,
    controller: Optional[BudgetController],
) -> np.ndarray:
    if compute_mask is None:
        mask = np.ones_like(risk_scores, dtype=bool)
    else:
        mask = compute_mask.astype(bool)
    if controller is None:
        return mask
    return controller.select(mask, risk_scores)

