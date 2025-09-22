from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np


def _safe_norm(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis)
    return np.maximum(n, eps)


def cosine_similarity_tokens(h_now_layers: List[np.ndarray], h_prev_layers: List[np.ndarray]) -> np.ndarray:
    """Compute per-token cosine between current and previous hidden, averaged over layers.

    Returns shape [seq].
    """
    assert len(h_now_layers) == len(h_prev_layers)
    num_layers = len(h_now_layers)
    seq_len = h_now_layers[0].shape[0]
    cos_sum = np.zeros((seq_len,), dtype=np.float32)
    for li in range(num_layers):
        a = h_now_layers[li]
        b = h_prev_layers[li]
        num = np.sum(a * b, axis=-1)
        den = _safe_norm(a, axis=-1) * _safe_norm(b, axis=-1)
        cos = num / den
        cos_sum += cos.astype(np.float32)
    return cos_sum / float(num_layers)


def delta_l2_ratio_tokens(h_now_layers: List[np.ndarray], h_prev_layers: List[np.ndarray], eps: float = 1e-8) -> np.ndarray:
    """Per-token L2 change ratio averaged over layers.

    Returns shape [seq].
    """
    assert len(h_now_layers) == len(h_prev_layers)
    num_layers = len(h_now_layers)
    seq_len = h_now_layers[0].shape[0]
    ratio_sum = np.zeros((seq_len,), dtype=np.float32)
    for li in range(num_layers):
        a = h_now_layers[li]
        b = h_prev_layers[li]
        diff = a - b
        num = _safe_norm(diff, axis=-1)
        den = _safe_norm(b, axis=-1) + eps
        ratio = num / den
        ratio_sum += ratio.astype(np.float32)
    return ratio_sum / float(num_layers)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def kl_divergence_tokens(logits_now: np.ndarray, logits_prev: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """KL(softmax now || softmax prev) per token.

    Returns shape [seq].
    """
    p = softmax(logits_now, axis=-1)
    q = softmax(logits_prev, axis=-1)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    kl = np.sum(p * (np.log(p) - np.log(q)), axis=-1)
    return kl.astype(np.float32)


def logits_entropy_and_margin(logits: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Return entropy and (p1 - p2) margin per token.

    Returns two arrays of shape [seq].
    """
    p = softmax(logits, axis=-1)
    p = np.clip(p, eps, 1.0)
    entropy = -np.sum(p * np.log(p), axis=-1)
    # margin
    sorted_p = np.sort(p, axis=-1)[:, ::-1]
    top1 = sorted_p[:, 0]
    top2 = sorted_p[:, 1] if sorted_p.shape[1] > 1 else np.zeros_like(top1)
    margin = top1 - top2
    return entropy.astype(np.float32), margin.astype(np.float32)


