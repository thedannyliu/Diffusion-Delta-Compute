from __future__ import annotations

import os
from typing import Optional

import numpy as np
def _try_import_matplotlib():
    try:
        import matplotlib  # type: ignore
        matplotlib.use("agg", force=True)
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:
        return None


def save_heatmap(
    arr: np.ndarray,
    path: str,
    title: Optional[str] = None,
    xlabel: str = "prompts",
    ylabel: str = "steps",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt = _try_import_matplotlib()
    if plt is None:
        # Fallback: save raw array
        np.save(path.replace(".png", ".npy"), arr)
        return
    plt.figure(figsize=(6, 4))
    plt.imshow(arr, aspect="auto", cmap="viridis")
    plt.colorbar(label="value")
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_hist(values: np.ndarray, path: str, bins: int = 50, title: Optional[str] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt = _try_import_matplotlib()
    if plt is None:
        # Fallback: save values as csv
        np.savetxt(path.replace(".png", ".csv"), values.reshape(-1, 1), delimiter=",", header="value", comments="")
        return
    plt.figure(figsize=(5, 3))
    plt.hist(values, bins=bins, color="steelblue", alpha=0.9)
    if title:
        plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_errorbar(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    path: str,
    title: Optional[str] = None,
    xlabel: str = "x",
    ylabel: str = "y",
    color: str = "steelblue",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt = _try_import_matplotlib()
    if plt is None:
        data = np.stack([x, y, yerr], axis=-1)
        np.savetxt(path.replace(".png", ".csv"), data, delimiter=",", header="x,mean,std", comments="")
        return
    plt.figure(figsize=(6, 4))
    plt.errorbar(x, y, yerr=yerr, fmt="o-", color=color, ecolor=color, elinewidth=1.2, capsize=4, capthick=1.2)
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _compute_roc_points(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    tps = np.cumsum(labels_sorted)
    fps = np.cumsum(1.0 - labels_sorted)
    pos = labels.sum()
    neg = labels.shape[0] - pos
    pos = max(pos, 1.0)
    neg = max(neg, 1.0)
    tpr = tps / pos
    fpr = fps / neg
    return np.stack([np.concatenate([[0.0], fpr, [1.0]]), np.concatenate([[0.0], tpr, [1.0]])], axis=-1)


def _compute_pr_points(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    tps = np.cumsum(labels_sorted)
    fps = np.cumsum(1.0 - labels_sorted)
    precision = tps / np.maximum(tps + fps, 1e-9)
    recall = tps / max(labels.sum(), 1e-9)
    return np.stack([np.concatenate([[0.0], recall, [1.0]]), np.concatenate([[precision[0]], precision, [labels.mean()]])], axis=-1)


def save_roc_curve(labels: np.ndarray, scores: np.ndarray, path: str, title: Optional[str] = None) -> None:
    plt = _try_import_matplotlib()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    curve = _compute_roc_points(labels.astype(np.float32), scores.astype(np.float32))
    if plt is None:
        np.savetxt(path.replace(".png", "_roc.csv"), curve, delimiter=",", header="fpr,tpr", comments="")
        return
    plt.figure(figsize=(4.5, 4.5))
    plt.plot(curve[:, 0], curve[:, 1], color="tab:red", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    if title:
        plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_pr_curve(labels: np.ndarray, scores: np.ndarray, path: str, title: Optional[str] = None) -> None:
    plt = _try_import_matplotlib()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    curve = _compute_pr_points(labels.astype(np.float32), scores.astype(np.float32))
    if plt is None:
        np.savetxt(path.replace(".png", "_pr.csv"), curve, delimiter=",", header="recall,precision", comments="")
        return
    plt.figure(figsize=(4.5, 4.5))
    plt.plot(curve[:, 0], curve[:, 1], color="tab:blue", linewidth=2)
    if title:
        plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_reliability_diagram(
    labels: np.ndarray,
    scores: np.ndarray,
    path: str,
    title: Optional[str] = None,
    num_bins: int = 10,
) -> None:
    plt = _try_import_matplotlib()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    probs = np.clip(scores.astype(np.float32), 1e-6, 1 - 1e-6)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    inds = np.digitize(probs, bins) - 1
    bin_true = []
    bin_pred = []
    for b in range(num_bins):
        mask = inds == b
        if not np.any(mask):
            bin_true.append(0.0)
            bin_pred.append((bins[b] + bins[b + 1]) / 2.0)
            continue
        bin_true.append(float(labels[mask].mean()))
        bin_pred.append(float(probs[mask].mean()))
    if plt is None:
        np.savetxt(path.replace(".png", "_calibration.csv"), np.stack([bin_pred, bin_true], axis=-1), delimiter=",", header="pred,true", comments="")
        return
    plt.figure(figsize=(4.5, 4.5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.plot(bin_pred, bin_true, marker="o", color="tab:purple", linewidth=2)
    if title:
        plt.title(title)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_tradeoff_curve(
    x: np.ndarray,
    y: np.ndarray,
    path: str,
    title: Optional[str] = None,
    xlabel: str = "coverage",
    ylabel: str = "risk",
    color: str = "tab:green",
) -> None:
    plt = _try_import_matplotlib()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if plt is None:
        np.savetxt(path.replace(".png", "_curve.csv"), np.stack([x, y], axis=-1), delimiter=",", header="x,y", comments="")
        return
    plt.figure(figsize=(5, 3.5))
    plt.plot(x, y, marker="o", linewidth=2, color=color)
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
