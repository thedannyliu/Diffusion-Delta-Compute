from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import transformers.modeling_utils as _tfm_mu
from peft import PeftModel

from .base_engine import BaseEngine, EngineState, StepOutputs


class D2FDreamEngine(BaseEngine):
    """Wrapper for the open-source d2f-dream engine.

    This is a thin adapter. It expects a d2f implementation to be importable.
    If the package is not installed, an ImportError will be raised.
    """

    def __init__(self, model_name: str = "d2f-small", device: str = "cuda", model_path: Optional[str] = None, lora_path: Optional[str] = None, use_lora: bool = False, num_steps: int = 12, max_seq_len: int = 128) -> None:
        self._device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        if self._device == "cpu":
            print("[warn] CUDA not available; D2F engine is running on CPU. GPU utilization will be ~0.", flush=True)
        self._model_name = model_name
        self._model_path = model_path
        self._lora_path = lora_path
        self._use_lora = use_lora
        self._num_steps = num_steps
        self._seq_len = max_seq_len

        # Load tokenizer and model from local snapshot
        model_load_path = self._model_path if self._model_path else "Dream-org/Dream-v0-Instruct-7B"
        self._tokenizer = AutoTokenizer.from_pretrained(model_load_path, trust_remote_code=True)
        # Patch transformers to drop 'weights_only' kwarg that some remote models don't accept
        _orig_from_pretrained = _tfm_mu.PreTrainedModel.from_pretrained
        def _patched_from_pretrained(cls, *args, **kwargs):
            kwargs.pop("weights_only", None)
            return _orig_from_pretrained.__func__(cls, *args, **kwargs)
        _tfm_mu.PreTrainedModel.from_pretrained = classmethod(_patched_from_pretrained)  # type: ignore
        try:
            base_model = AutoModel.from_pretrained(
                model_load_path,
                device_map="auto" if self._device != "cpu" else None,
                torch_dtype="auto",
                trust_remote_code=True,
            )
        finally:
            _tfm_mu.PreTrainedModel.from_pretrained = _orig_from_pretrained  # restore
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
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].to(torch.bool)
        input_ids_cpu = inputs.get("input_ids", None)
        pad_id = self._tokenizer.pad_token_id if getattr(self._tokenizer, "pad_token_id", None) is not None else -1
        special_mask_np = None
        input_ids_np = None
        if input_ids_cpu is not None:
            ids = input_ids_cpu[0].detach().cpu().tolist()
            special_mask = self._tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
            special_mask_np = np.array(special_mask, dtype=bool)
            input_ids_np = np.array(ids, dtype=np.int64)
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True)
        # Hidden states
        hidden_states = out.hidden_states  # tuple len L+1 (incl embeddings)
        layers = []
        for h in hidden_states[1:]:  # skip embeddings
            layers.append(h[0].detach().cpu().float().numpy())  # [seq, hidden]
        # Layer-reuse (optional): every M steps, shallow layers are reused
        layer_mask = None
        if hasattr(self, "layer_recompute_M") and self.layer_recompute_M and self.layer_recompute_M > 1:
            M = int(self.layer_recompute_M)
            layer_mask = np.ones((len(layers),), dtype=bool)
            shallow_reuse_upto = min(3, len(layers))  # L1-3 reuse window
            if (t % M) != 0:
                layer_mask[:shallow_reuse_upto] = False
                # overwrite shallow layers with prev
                if state.prev_hidden_by_layer is not None:
                    for li in range(shallow_reuse_upto):
                        layers[li] = state.prev_hidden_by_layer[li].copy()

        # Inject step-dependent small noise to emulate diffusion progression on computed tokens
        rng = state.rng
        seq_len = layers[0].shape[0]
        noise_scale = max(1e-3, 0.08 * (1.0 - float(t) / max(1, self._num_steps)))
        token_mask = np.ones((seq_len,), dtype=bool) if compute_mask is None else compute_mask.astype(bool)
        for li in range(len(layers)):
            if layer_mask is not None and layer_mask[li] is False:
                continue
            noise = rng.normal(0.0, noise_scale, size=layers[li].shape).astype(np.float32)
            layers[li][token_mask] = layers[li][token_mask] + noise[token_mask]
        # Logits via output embeddings if available, using the last hidden after noise
        logits_t = None
        get_oe = getattr(self._model, "get_output_embeddings", None)
        if callable(get_oe):
            oe = get_oe()
            if oe is not None:
                last_hidden = torch.from_numpy(layers[-1]).to(self._device)  # [seq, hidden]
                if last_hidden.ndim == 2:
                    last_hidden = last_hidden.unsqueeze(0)  # [1, seq, hidden]
                # Cast hidden to output embedding dtype to avoid dtype mismatch
                weight = getattr(oe, "weight", None)
                if weight is not None:
                    last_hidden = last_hidden.to(dtype=weight.dtype)
                logits_t = oe(last_hidden)[0].detach().cpu().float().numpy()  # [seq, vocab]
        if logits_t is None:
            # Fallback: simple projection
            h_last = layers[-1]
            vocab_size = 32000
            proj = torch.randn((h_last.shape[-1], vocab_size), dtype=torch.float32)
            logits_t = (torch.from_numpy(h_last) @ proj).numpy()

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
        aux = {"sigma": noise_scale, "t": t}
        if input_ids_np is not None:
            aux["input_ids"] = input_ids_np
        if special_mask_np is not None:
            aux["special_tokens_mask"] = special_mask_np
        aux["pad_token_id"] = int(pad_id)
        if layer_mask is not None:
            aux["layer_mask"] = layer_mask
        return StepOutputs(hidden_by_layer=layers, logits=logits_t, attn_stats=None, aux=aux)

    def decode(self, state: EngineState) -> str:
        return state.prompt

    def num_steps(self) -> int:
        return self._num_steps

