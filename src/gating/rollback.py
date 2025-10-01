from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.engine.base_engine import EngineState


@dataclass
class RollbackSnapshot:
    hidden_by_layer: Optional[List[np.ndarray]]
    logits: Optional[np.ndarray]
    step_index: int


class RollbackBuffer:
    """Simple rollback buffer that stores the last pre-step state."""

    def __init__(self) -> None:
        self._snapshot: Optional[RollbackSnapshot] = None

    def save(self, state: EngineState) -> None:
        hidden = None if state.prev_hidden_by_layer is None else [h.copy() for h in state.prev_hidden_by_layer]
        logits = None if state.prev_logits is None else state.prev_logits.copy()
        self._snapshot = RollbackSnapshot(hidden_by_layer=hidden, logits=logits, step_index=state.step_index)

    def restore(self, state: EngineState) -> None:
        if self._snapshot is None:
            return
        state.prev_hidden_by_layer = None if self._snapshot.hidden_by_layer is None else [h.copy() for h in self._snapshot.hidden_by_layer]
        state.prev_logits = None if self._snapshot.logits is None else self._snapshot.logits.copy()
        state.step_index = self._snapshot.step_index

    def clear(self) -> None:
        self._snapshot = None

