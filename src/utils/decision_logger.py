from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Iterable, Optional, Sequence


class DecisionLogger:
    """Utility for streaming per-token decision logs and final outputs."""

    def __init__(self, run_dir: str, mode: str, run_id: str) -> None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        decisions_dir = os.path.join(run_dir, "runs", "decisions")
        os.makedirs(decisions_dir, exist_ok=True)
        self._decision_path = os.path.join(decisions_dir, f"{mode}_decisions_{timestamp}.jsonl")
        self._final_path = os.path.join(decisions_dir, f"{mode}_final_outputs_{timestamp}.jsonl")
        self._decision_file = open(self._decision_path, "w", encoding="utf-8")
        self._final_file = open(self._final_path, "w", encoding="utf-8")
        self._lock = threading.Lock()
        self.run_id = run_id
        self.mode = mode

    @property
    def decision_path(self) -> str:
        return self._decision_path

    @property
    def final_outputs_path(self) -> str:
        return self._final_path

    def log_decision(
        self,
        *,
        prompt_id: int,
        step: int,
        token_idx: int,
        decision: str,
        k_applied: int,
        risk: float,
        risk_threshold: float,
        margin: float,
        kl: float,
        one_minus_cos: float,
        delta_l2: float,
        delta_l2_norm: float,
        sigma: float,
        layer_band: str,
        profile: Optional[str],
        extras: Optional[dict] = None,
    ) -> None:
        row = {
            "run_id": self.run_id,
            "mode": self.mode,
            "prompt_id": int(prompt_id),
            "step": int(step),
            "token_idx": int(token_idx),
            "decision": decision,
            "k_applied": int(k_applied),
            "risk": float(risk),
            "risk_threshold": float(risk_threshold),
            "margin": float(margin),
            "kl": float(kl),
            "one_minus_cos": float(one_minus_cos),
            "delta_l2": float(delta_l2),
            "delta_l2_norm": float(delta_l2_norm),
            "sigma": float(sigma),
            "layer_band": layer_band,
        }
        if profile is not None:
            row["profile"] = profile
        if extras:
            for key, value in extras.items():
                row[key] = value
        with self._lock:
            self._decision_file.write(json.dumps(row) + "\n")

    def log_final_output(
        self,
        *,
        prompt_id: int,
        tokens: Sequence[int],
        margins: Sequence[float],
        logits_checksum: Optional[float] = None,
        extras: Optional[dict] = None,
    ) -> None:
        row = {
            "run_id": self.run_id,
            "mode": self.mode,
            "prompt_id": int(prompt_id),
            "tokens": [int(t) for t in tokens],
            "margins": [float(m) for m in margins],
        }
        if logits_checksum is not None:
            row["logits_checksum"] = float(logits_checksum)
        if extras:
            for key, value in extras.items():
                row[key] = value
        with self._lock:
            self._final_file.write(json.dumps(row) + "\n")

    def close(self) -> None:
        with self._lock:
            try:
                self._decision_file.close()
            finally:
                self._final_file.close()


def close_loggers(loggers: Iterable[DecisionLogger]) -> None:
    for logger in loggers:
        try:
            logger.close()
        except Exception:
            pass

