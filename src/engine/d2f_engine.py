from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .base_engine import BaseEngine, EngineState, StepOutputs


class D2FDreamEngine(BaseEngine):
    """Wrapper for the open-source d2f-dream engine.

    This is a thin adapter. It expects a d2f implementation to be importable.
    If the package is not installed, an ImportError will be raised.
    """

    def __init__(self, model_name: str = "d2f-small", device: str = "cuda", model_path: Optional[str] = None, lora_path: Optional[str] = None, use_lora: bool = False) -> None:
        try:
            # Placeholder import paths – update to actual package as needed.
            import d2f_dream as d2f  # type: ignore
        except Exception as e:
            raise ImportError(
                "d2f-dream is not installed. Please install the engine and set PYTHONPATH or use requirements.txt instructions."
            ) from e

        self._d2f = d2f
        self._device = device
        # Construct model/engine according to d2f API (pseudo-code)
        # self._engine = d2f.load_engine(model_name=model_name, device=device)
        # For PoC placeholder, we just store name
        self._model_name = model_name
        self._model_path = model_path
        self._lora_path = lora_path
        self._use_lora = use_lora
        self._num_steps = 12
        self._vocab_size = 32000
        self._hidden_size = 1024
        self._num_layers = 24
        self._seq_len = 128

    def encode_prompt(self, prompt: str) -> EngineState:
        rng = np.random.default_rng(1234 + hash(prompt) % 10000)
        return EngineState(prompt=prompt, step_index=0, rng=rng)

    def step(self, state: EngineState, t: int, compute_mask: Optional[np.ndarray] = None) -> StepOutputs:
        # TODO: Replace with calls into real d2f engine per-step API with compute_mask support
        # For now, use a deterministic placeholder evolution so the harness runs end-to-end.
        rng = state.rng
        if state.prev_hidden_by_layer is None:
            hidden_by_layer = [
                rng.normal(0.0, 1.0, size=(self._seq_len, self._hidden_size)).astype(np.float32)
                for _ in range(self._num_layers)
            ]
        else:
            hidden_by_layer = [h.copy() for h in state.prev_hidden_by_layer]
            noise = [rng.normal(0.0, 0.1, size=h.shape).astype(np.float32) for h in hidden_by_layer]
            hidden_by_layer = [h + n for h, n in zip(hidden_by_layer, noise)]
            if compute_mask is not None:
                mask = compute_mask.astype(bool)
                hidden_by_layer = [np.where(mask[:, None], h, state.prev_hidden_by_layer[i]) for i, h in enumerate(hidden_by_layer)]

        # Simple logits projection for placeholder
        W = np.random.default_rng(2024).normal(0.0, 0.1, size=(self._hidden_size, self._vocab_size)).astype(np.float32)
        pooled = sum(hidden_by_layer) / float(len(hidden_by_layer))
        logits = pooled @ W

        state.prev_hidden_by_layer = [h.copy() for h in hidden_by_layer]
        state.prev_logits = logits.copy()
        state.step_index = t

        return StepOutputs(hidden_by_layer=hidden_by_layer, logits=logits, attn_stats=None, aux={})

    def decode(self, state: EngineState) -> str:
        return "<decoded text>"

    def num_steps(self) -> int:
        return self._num_steps


