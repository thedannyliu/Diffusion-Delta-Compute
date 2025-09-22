from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # Optional dependency


@dataclass
class ExperimentPaths:
    data_root: str
    models_root: str
    outputs_root: str


def load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if yaml is None:
        raise ImportError("PyYAML not installed; pip install pyyaml or avoid --config")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_output_dirs(outputs_root: str, phase: str, mode: str, engine: str, exp_name: Optional[str]) -> str:
    run_dir = os.path.join(outputs_root, phase, mode, engine, (exp_name or "run") + "_" + os.popen("date +%Y%m%d_%H%M%S").read().strip())
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "runs"), exist_ok=True)
    return run_dir


def save_manifest(run_dir: str, manifest: Dict[str, Any]) -> None:
    path = os.path.join(run_dir, "run_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


