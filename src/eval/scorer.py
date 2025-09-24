from __future__ import annotations

from typing import List

import numpy as np


def perplexity_from_nll(nlls: List[float]) -> float:
    nll = float(np.mean(np.array(nlls, dtype=np.float64)))
    return float(np.exp(nll))


def exact_match(pred: str, gold: str) -> bool:
    return pred.strip() == gold.strip()


def accuracy(preds: List[str], labels: List[str]) -> float:
    correct = sum(1 for p, g in zip(preds, labels) if exact_match(p, g))
    return correct / max(1, len(labels))


