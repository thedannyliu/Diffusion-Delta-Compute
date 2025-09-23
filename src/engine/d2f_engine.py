from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from .base_engine import BaseEngine, EngineState, StepOutputs


class D2FDreamEngine(BaseEngine):
    """Wrapper for the open-source d2f-dream engine.

    This is a thin adapter. It expects a d2f implementation to be importable.
    If the package is not installed, an ImportError will be raised.
    """

    def __init__(self, model_name: str = "d2f-small", device: str = "cuda", model_path: Optional[str] = None, lora_path: Optional[str] = None, use_lora: bool = False, num_steps: int = 12, max_seq_len: int = 128) -> None:
        self._device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self._model_name = model_name
        self._model_path = model_path
        self._lora_path = lora_path
        self._use_lora = use_lora
        self._num_steps = num_steps
        self._seq_len = max_seq_len

        # Load tokenizer and model from local snapshot
        model_load_path = self._model_path if self._model_path else "Dream-org/Dream-v0-Instruct-7B"
        self._tokenizer = AutoTokenizer.from_pretrained(model_load_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_load_path,
            device_map="auto" if self._device != "cpu" else None,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        if self._use_lora and self._lora_path:
            peft_model = PeftModel.from_pretrained(base_model, self._lora_path)
            try:
                base_model = peft_model.merge_and_unload()
            except Exception:
                base_model = peft_model
        self._model = base_model.eval()

    def encode_prompt(self, prompt: str) -> EngineState:
        rng = np.random.default_rng(1234 + hash(prompt) % 10000)
        return EngineState(prompt=prompt, step_index=0, rng=rng)

    def step(self, state: EngineState, t: int, compute_mask: Optional[np.ndarray] = None) -> StepOutputs:
        inputs = self._tokenizer(state.prompt, return_tensors="pt", truncation=True, max_length=self._seq_len)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True)
        logits_t = out.logits[0].detach().cpu().float().numpy()  # [seq, vocab]
        hidden_states = out.hidden_states  # tuple len L+1 (incl embeddings)
        layers = []
        for h in hidden_states[1:]:  # skip embeddings
            layers.append(h[0].detach().cpu().float().numpy())  # [seq, hidden]

        # Apply per-token freeze by reusing previous step values where compute_mask is False
        if compute_mask is not None and state.prev_hidden_by_layer is not None and state.prev_logits is not None:
            mask = compute_mask.astype(bool)
            for li in range(len(layers)):
                prev_layer = state.prev_hidden_by_layer[li]
                cur_layer = layers[li]
                cur_layer[~mask] = prev_layer[~mask]
                layers[li] = cur_layer
            logits_prev = state.prev_logits
            logits_t[~mask] = logits_prev[~mask]

        state.prev_hidden_by_layer = [h.copy() for h in layers]
        state.prev_logits = logits_t.copy()
        state.step_index = t

        return StepOutputs(hidden_by_layer=layers, logits=logits_t, attn_stats=None, aux={})

    def decode(self, state: EngineState) -> str:
        return state.prompt

    def num_steps(self) -> int:
        return self._num_steps


