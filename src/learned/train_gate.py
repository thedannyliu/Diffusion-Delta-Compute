from __future__ import annotations
import argparse
from typing import List, Tuple

import numpy as np


def load_feature_arrays(paths: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    arrays = []
    labels = []
    feature_names = None
    for path in paths:
        data = np.load(path, allow_pickle=True)
        X = data["features"] if "features" in data else data["X"]
        y = data["labels"] if "labels" in data else data["y"]
        if feature_names is None and "feature_names" in data:
            feature_names = list(data["feature_names"])
        arrays.append(X)
        labels.append(y)
    X_all = np.concatenate(arrays, axis=0)
    y_all = np.concatenate(labels, axis=0)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X_all.shape[1])]
    return X_all, y_all, feature_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--threshold_grid", type=str, default=None)
    args = parser.parse_args()

    X, y, names = load_feature_arrays(args.features)
    # Placeholder: save a dummy model file to unblock pipeline
    np.savez(args.out, coef=np.zeros((X.shape[1],), dtype=np.float32), names=np.array(names, dtype=object))
    print(f"Saved dummy learned gate weights to {args.out}")


if __name__ == "__main__":
    main()



import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight learned gate")
    parser.add_argument("--features", type=str, nargs="+", help="Paths or glob patterns to teacher feature .npz files")
    parser.add_argument("--out", type=str, required=True, help="Output path for learned gate weights (.npz)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--pos_weight", type=float, default=1.0, help="Positive class weight for BCE loss")
    parser.add_argument("--threshold_grid", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_paths(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pat in patterns:
        p = Path(pat)
        if p.exists():
            if p.is_dir():
                paths.extend(sorted(p.glob("*.npz")))
            else:
                paths.append(p)
            continue
        matched = list(Path().glob(pat))
        if matched:
            paths.extend(sorted(matched))
    unique_paths = []
    seen = set()
    for p in paths:
        if p not in seen:
            unique_paths.append(p)
            seen.add(p)
    return unique_paths


def load_feature_arrays(paths: Sequence[Path]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feats = []
    labels = []
    feature_names: List[str] = []
    for path in paths:
        data = np.load(path, allow_pickle=False)
        feats.append(data["features"].astype(np.float32))
        labels.append(data["labels"].astype(np.float32))
        if "feature_names" in data and not feature_names:
            feature_names = [str(x) for x in data["feature_names"]]
    if not feats:
        raise ValueError("No feature files found")
    features = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0)
    if not feature_names:
        feature_names = [f"f{i}" for i in range(features.shape[1])]
    return features, y, feature_names


def train_linear_gate(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    pos_weight: float,
    seed: int,
) -> Tuple[nn.Module, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0.0] = 1.0
    norm_features = (features - mean) / std

    X = torch.from_numpy(norm_features.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.float32))

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = nn.Linear(X.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    return model.eval(), mean, std


def evaluate_thresholds(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    thresholds: Sequence[float],
) -> Tuple[float, dict]:
    with torch.no_grad():
        norm = (features - mean) / std
        logits = model(torch.from_numpy(norm.astype(np.float32))).squeeze(-1).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
    best_threshold = thresholds[0]
    best_metric = -1.0
    metrics = {}
    for th in thresholds:
        preds = (probs >= th).astype(np.float32)
        tp = float(np.sum((preds == 1.0) & (labels == 1.0)))
        fp = float(np.sum((preds == 1.0) & (labels == 0.0)))
        fn = float(np.sum((preds == 0.0) & (labels == 1.0)))
        tn = float(np.sum((preds == 0.0) & (labels == 0.0)))
        precision = tp / max(1.0, (tp + fp))
        recall = tp / max(1.0, (tp + fn))
        f1 = 2 * precision * recall / max(1e-8, (precision + recall))
        acc = (tp + tn) / max(1.0, len(labels))
        metrics[th] = {"precision": precision, "recall": recall, "f1": f1, "acc": acc}
        if f1 > best_metric:
            best_metric = f1
            best_threshold = th
    return best_threshold, {"threshold_metrics": metrics}


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args.features)
    features, labels, feature_names = load_feature_arrays(paths)

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(features.shape[0])
    split = int(features.shape[0] * (1.0 - args.val_ratio))
    train_idx = indices[:split]
    val_idx = indices[split:]
    if len(val_idx) == 0:
        val_idx = train_idx

    model, mean, std = train_linear_gate(
        features[train_idx],
        labels[train_idx],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pos_weight=args.pos_weight,
        seed=args.seed,
    )

    threshold_candidates = [float(x) for x in args.threshold_grid.split(",") if x]
    best_threshold, eval_details = evaluate_thresholds(
        model,
        features[val_idx],
        labels[val_idx],
        mean,
        std,
        threshold_candidates,
    )

    weights = model.weight.detach().cpu().numpy().astype(np.float32).squeeze(0)
    bias = model.bias.detach().cpu().numpy().astype(np.float32).squeeze(0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        weights=weights,
        bias=bias,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        threshold=np.array([best_threshold], dtype=np.float32),
        feature_names=np.array(feature_names),
        meta=json.dumps({"train_examples": int(train_idx.shape[0]), "val_examples": int(val_idx.shape[0])}),
        eval=json.dumps(eval_details),
    )

    print(f"[learned_gate] Saved weights to {out_path} (threshold={best_threshold:.3f})")


if __name__ == "__main__":
    main()
