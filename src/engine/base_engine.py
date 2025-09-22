from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

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
    aux: Optional[Dict] = None


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


