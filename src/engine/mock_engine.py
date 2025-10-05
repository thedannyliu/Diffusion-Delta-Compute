from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .base_engine import BaseEngine, EngineState, StepOutputs


class MockDiffusionEngine(BaseEngine):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        hidden_size: int,
        sequence_length: int,
        num_steps: int,
        rng_seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self._num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self._num_steps = num_steps
        self.rng_seed = rng_seed

    def encode_prompt(self, prompt: str) -> EngineState:
        rng = np.random.default_rng(self.rng_seed + hash(prompt) % 10000)
        return EngineState(prompt=prompt, step_index=0, rng=rng)

    def _sample_hidden(self, rng: np.random.Generator) -> List[np.ndarray]:
        hidden = [
            rng.normal(loc=0.0, scale=1.0, size=(self.sequence_length, self.hidden_size)).astype(np.float32)
            for _ in range(self._num_layers)
        ]
        return hidden

    def _transition(self, prev: List[np.ndarray], rng: np.random.Generator, noise_scale: float) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for layer_hidden in prev:
            noise = rng.normal(loc=0.0, scale=noise_scale, size=layer_hidden.shape).astype(np.float32)
            out.append(layer_hidden + noise)
        return out

    def _logits_from_hidden(self, hidden_by_layer: List[np.ndarray]) -> np.ndarray:
        pooled = sum([h for h in hidden_by_layer]) / float(len(hidden_by_layer))
        # Cheap linear projection to vocab via fixed random matrix per instance
        rng = np.random.default_rng(12345)
        W = rng.normal(0.0, 0.1, size=(self.hidden_size, self.vocab_size)).astype(np.float32)
        logits = pooled @ W
        return logits

    def step(self, state: EngineState, t: int, compute_mask: Optional[np.ndarray] = None) -> StepOutputs:
        rng = state.rng
        if state.prev_hidden_by_layer is None:
            hidden_now = self._sample_hidden(rng)
        else:
            # Noise scale decays over steps to mimic convergence
            noise_scale = max(0.05, 1.0 - 0.08 * t)
            hidden_now = self._transition(state.prev_hidden_by_layer, rng, noise_scale=noise_scale)

        if compute_mask is not None and state.prev_hidden_by_layer is not None:
            # Only update tokens where compute_mask is True; others reuse previous
            mask = compute_mask.astype(bool)
            for li in range(len(hidden_now)):
                prev_layer = state.prev_hidden_by_layer[li]
                cur_layer = hidden_now[li]
                cur_layer[~mask] = prev_layer[~mask]
                hidden_now[li] = cur_layer

        logits = self._logits_from_hidden(hidden_now)

        # Update state
        state.prev_hidden_by_layer = [h.copy() for h in hidden_now]
        state.prev_logits = logits.copy()
        state.step_index = t

        return StepOutputs(hidden_by_layer=hidden_now, logits=logits, attn_stats=None, aux={})

    def decode(self, state: EngineState) -> str:
        return "<decoded text>"

    def num_steps(self) -> int:
        return self._num_steps

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        return " ".join(f"<tok{tid}>" for tid in token_ids)

    def greedy_generate(self, prompt: str, max_new_tokens: int = 16) -> List[int]:
        # Deterministic pseudo-generation: return ascending token ids capped by vocab
        return [min(self.vocab_size - 1, i % self.vocab_size) for i in range(max_new_tokens)]
