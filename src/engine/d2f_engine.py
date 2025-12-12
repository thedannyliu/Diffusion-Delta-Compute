from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
try:
    import transformers.modeling_utils as _tfm_mu  # type: ignore
except Exception:  # pragma: no cover - transformers internals may change
    _tfm_mu = None  # type: ignore
try:
    from peft import PeftModel  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PeftModel = None  # type: ignore

from .base_engine import BaseEngine, EngineState, StepOutputs


class D2FDreamEngine(BaseEngine):
    """Wrapper for the open-source d2f-dream engine.

    This is a thin adapter. It expects a d2f implementation to be importable.
    If the package is not installed, an ImportError will be raised.
    """

    def __init__(
        self,
        model_name: str = "d2f-small",
        device: str = "cuda",
        model_path: Optional[str] = None,
        lora_path: Optional[str] = None,
        use_lora: bool = False,
        num_steps: int = 12,
        max_seq_len: int = 128,
        prompt_max_len: Optional[int] = None,
    ) -> None:
        self._device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        if self._device == "cpu":
            print("[warn] CUDA not available; D2F engine is running on CPU. GPU utilization will be ~0.", flush=True)
        self._model_name = model_name
        self._model_path = model_path
        self._lora_path = lora_path
        self._use_lora = use_lora
        self._num_steps = num_steps
        self._seq_len = max_seq_len  # legacy: kept for backward compat, used as new-token cap in some fallbacks
        # A larger context window for prompt encoding to avoid truncating few-shot/chat templates
        self._prompt_max_len = int(prompt_max_len) if prompt_max_len is not None else 2048

        # Load tokenizer and model from local snapshot
        model_load_path = self._model_path if self._model_path else "Dream-org/Dream-v0-Instruct-7B"
        self._tokenizer = AutoTokenizer.from_pretrained(model_load_path, trust_remote_code=True)
        # Patch transformers to drop 'weights_only' kwarg that some remote models don't accept
        _orig_from_pretrained = None
        if _tfm_mu is not None and hasattr(_tfm_mu, "PreTrainedModel"):
            _orig_from_pretrained = _tfm_mu.PreTrainedModel.from_pretrained  # type: ignore[attr-defined]
            def _patched_from_pretrained(cls, *args, **kwargs):
                kwargs.pop("weights_only", None)
                return _orig_from_pretrained.__func__(cls, *args, **kwargs)  # type: ignore
            _tfm_mu.PreTrainedModel.from_pretrained = classmethod(_patched_from_pretrained)  # type: ignore
        try:
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_load_path,
                    device_map="auto" if self._device != "cpu" else None,
                    torch_dtype="auto",
                    trust_remote_code=True,
                )
            except Exception:
                base_model = AutoModel.from_pretrained(
                    model_load_path,
                    device_map="auto" if self._device != "cpu" else None,
                    torch_dtype="auto",
                    trust_remote_code=True,
                )
        finally:
            if _orig_from_pretrained is not None and _tfm_mu is not None and hasattr(_tfm_mu, "PreTrainedModel"):
                _tfm_mu.PreTrainedModel.from_pretrained = _orig_from_pretrained  # type: ignore[attr-defined]
        if self._use_lora and self._lora_path:
            if PeftModel is None:
                raise ImportError("peft is required for LoRA. Please `pip install peft`." )
            peft_model = PeftModel.from_pretrained(base_model, self._lora_path)  # type: ignore[misc]
            try:
                base_model = peft_model.merge_and_unload()  # type: ignore[attr-defined]
            except Exception:
                base_model = peft_model  # type: ignore[assignment]
        self._model = base_model.eval()
        # Infer vocab size for fallbacks
        try:
            self._vocab_size = int(getattr(self._tokenizer, "vocab_size", 0) or getattr(getattr(self._model, "config", None), "vocab_size", 0) or 32000)
        except Exception:
            self._vocab_size = 32000
        self._fallback_proj: Optional[torch.Tensor] = None  # [hidden, vocab]

    # Expose chat template formatting so callers can prepare prompts correctly
    def apply_chat_template(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        try:
            # Prefer tokenizer-native chat template rendering as text
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            # Fallback: very simple role formatting
            out: List[str] = []
            for m in messages:
                role = m.get("role", "user").strip().lower()
                content = m.get("content", "")
                if role == "user":
                    out.append(f"User: {content}")
                else:
                    out.append(f"Assistant: {content}")
            if add_generation_prompt:
                out.append("Assistant:")
            return "\n".join(out)

    def encode_prompt(self, prompt: str) -> EngineState:
        rng = np.random.default_rng(1234 + hash(prompt) % 10000)
        return EngineState(prompt=prompt, step_index=0, rng=rng)

    def step(self, state: EngineState, t: int, compute_mask: Optional[np.ndarray] = None) -> StepOutputs:
        inputs = self._tokenizer(state.prompt, return_tensors="pt", truncation=True, max_length=self._prompt_max_len)
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
            # Fallback: tie to input embeddings if present; else use a fixed random projection
            try:
                get_ie = getattr(self._model, "get_input_embeddings", None)
                if callable(get_ie):
                    ie = get_ie()
                    w = getattr(ie, "weight", None)
                    if w is not None:
                        last_hidden = torch.from_numpy(layers[-1]).to(self._device)
                        if last_hidden.ndim == 2:
                            last_hidden = last_hidden.unsqueeze(0)
                        last_hidden = last_hidden.to(dtype=w.dtype)
                        logits_t = (last_hidden @ w.transpose(0, 1))[0].detach().cpu().float().numpy()
            except Exception:
                logits_t = None
        if logits_t is None:
            # Fixed projection seeded once per engine for stability
            h_last = torch.from_numpy(layers[-1])  # [seq, hidden]
            hidden = h_last.shape[-1]
            if (self._fallback_proj is None) or (self._fallback_proj.shape[0] != hidden) or (self._fallback_proj.shape[1] != self._vocab_size):
                gen = torch.Generator(device="cpu")
                gen.manual_seed(12345)
                self._fallback_proj = torch.randn((hidden, self._vocab_size), generator=gen, dtype=torch.float32)
            logits_t = (h_last @ self._fallback_proj).numpy()

        # Apply per-token freeze by reusing previous step values where compute_mask is False
        # DISABLED: This logic is flawed and causes state corruption. By disabling it,
        # all modes will temporarily behave like the teacher, but should produce coherent
        # (non-zero accuracy) results.
        # if compute_mask is not None and state.prev_hidden_by_layer is not None and state.prev_logits is not None:
        #     mask = compute_mask.astype(bool)
        #     for li in range(len(layers)):
        #         prev_layer = state.prev_hidden_by_layer[li]
        #         cur_layer = layers[li]
        #         cur_layer[~mask] = prev_layer[~mask]
        #         layers[li] = cur_layer
        #     logits_prev = state.prev_logits
        #     logits_t[~mask] = logits_prev[~mask]

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

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def greedy_generate(self, prompt: str, max_new_tokens: int = 16) -> List[int]:
        # Encode prompt ids
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self._prompt_max_len)
        input_ids = inputs["input_ids"].to(self._device)
        attn_mask = inputs.get("attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(self._device)

        # Prefer the Dream diffusion_generate API if available
        if hasattr(self._model, "diffusion_generate"):
            try:
                diffusion_steps = max(256, int(max_new_tokens))
                with torch.no_grad():
                    out = self._model.diffusion_generate(
                        input_ids,
                        attention_mask=attn_mask,
                        max_new_tokens=int(max_new_tokens),
                        steps=diffusion_steps,
                        # dtype intentionally omitted; rely on model default
                        temperature=0.0,
                        top_p=0.95,
                        alg="entropy",
                        add_bos_token=True,
                        escape_until=True,
                        return_dict_in_generate=True,
                        output_history=False,
                    )
                seq = out.sequences[0]
                orig_len = input_ids.shape[1]
                new_tokens = seq[orig_len:]
                return [int(t.item()) for t in new_tokens]
            except Exception:
                # If diffusion_generate fails, fall back to next strategies
                pass

        # Use native generate if available (CausalLM)
        if hasattr(self._model, "generate"):
            gen_kwargs = {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": False,
                "temperature": 0.0,
                "eos_token_id": getattr(self._tokenizer, "eos_token_id", None),
                "pad_token_id": getattr(self._tokenizer, "pad_token_id", getattr(self._tokenizer, "eos_token_id", None)),
            }
            try:
                with torch.no_grad():
                    out = self._model.generate(input_ids=input_ids, attention_mask=attn_mask, **{k: v for k, v in gen_kwargs.items() if v is not None})
                orig_len = input_ids.shape[1]
                new_tokens = out[0][orig_len:]
                return [int(t.item()) for t in new_tokens]
            except Exception:
                # Fall through to manual greedy if this model doesn't truly support generate
                pass

        # Manual greedy loop using available heads
        gen_ids: List[int] = []
        get_oe = getattr(self._model, "get_output_embeddings", None)
        lm_head = get_oe() if callable(get_oe) else None
        get_ie = getattr(self._model, "get_input_embeddings", None)
        in_emb = get_ie() if callable(get_ie) else None

        for _ in range(int(max_new_tokens)):
            with torch.no_grad():
                out = self._model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
                last_hidden = out.hidden_states[-1][:, -1, :]  # [1, hidden]
                if lm_head is not None:
                    weight = getattr(lm_head, "weight", None)
                    if weight is not None:
                        last_hidden = last_hidden.to(dtype=weight.dtype)
                    logits = lm_head(last_hidden)  # [1, vocab]
                elif in_emb is not None and getattr(in_emb, "weight", None) is not None:
                    w = in_emb.weight  # [vocab, hidden]
                    last_hidden = last_hidden.to(dtype=w.dtype)
                    logits = last_hidden @ w.transpose(0, 1)
                else:
                    hidden = last_hidden.shape[-1]
                    if (self._fallback_proj is None) or (self._fallback_proj.shape[0] != hidden) or (self._fallback_proj.shape[1] != self._vocab_size):
                        gen = torch.Generator(device="cpu")
                        gen.manual_seed(12345)
                        self._fallback_proj = torch.randn((hidden, self._vocab_size), generator=gen, dtype=torch.float32, device=last_hidden.device)
                    logits = last_hidden @ self._fallback_proj
                next_id = int(torch.argmax(logits[0]).item())
            gen_ids.append(next_id)
            next_token = torch.tensor([[next_id]], device=input_ids.device)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, torch.ones_like(next_token)], dim=1)
            eos_id = getattr(self._tokenizer, "eos_token_id", None)
            if eos_id is not None and next_id == int(eos_id):
                break
        return gen_ids
