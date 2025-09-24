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


def save_heatmap(arr: np.ndarray, path: str, title: Optional[str] = None) -> None:
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
    plt.xlabel("prompts")
    plt.ylabel("steps")
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


