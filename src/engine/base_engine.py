from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class EngineState:
    prompt: str
    step_index: int
    rng: np.random.Generator
    prev_hidden_by_layer: Optional[List[np.ndarray]] = None
    prev_logits: Optional[np.ndarray] = None


@dataclass
class StepOutputs:
    hidden_by_layer: List[np.ndarray]  # [n_layers][seq, hidden]
    logits: np.ndarray  # [seq, vocab]
    attn_stats: Optional[Dict] = None
    aux: Optional[Dict] = None  # may include: input_ids, special_tokens_mask, pad_token_id, layer_mask, sigma, t


class BaseEngine(ABC):
    @abstractmethod
    def encode_prompt(self, prompt: str) -> EngineState:
        raise NotImplementedError

    @abstractmethod
    def step(self, state: EngineState, t: int, compute_mask: Optional[np.ndarray] = None) -> StepOutputs:
        raise NotImplementedError

    @abstractmethod
    def decode(self, state: EngineState) -> str:
        raise NotImplementedError

    @abstractmethod
    def num_steps(self) -> int:
        raise NotImplementedError

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        """Decode a sequence of token ids into text (default: raises)."""
        raise NotImplementedError("decode_tokens not implemented for this engine")

    def decode_token(self, token_id: int) -> str:
        """Decode a single token id into text."""
        return self.decode_tokens([token_id])

    def greedy_generate(self, prompt: str, max_new_tokens: int = 16) -> List[int]:
        """Greedy-generate token ids conditioned on the prompt.

        Default implementation is not available; engines that support generation
        (e.g., d2f backed by a causal LM head) should override this.
        """
        raise NotImplementedError("greedy_generate not implemented for this engine")
