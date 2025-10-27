import argparse
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.utils.logging import init_run_logging
from src.utils.timer import Timer
from src.utils.decision_logger import DecisionLogger, close_loggers
from src.engine.base_engine import BaseEngine
from src.engine.mock_engine import MockDiffusionEngine

try:  # Optional dependency, only needed when using the d2f backend
    from src.engine.d2f_engine import D2FDreamEngine
except Exception as _d2f_import_exc:  # pragma: no cover - lazy import guard
    # Surface real import errors to logs so SLURM users can diagnose
    try:
        import sys, traceback
        print(f"[d2f] Import failed: {_d2f_import_exc}", file=sys.stderr)
        traceback.print_exc()
    except Exception:
        pass
    D2FDreamEngine = None
from src.probes.metrics import (
    cosine_similarity_tokens,
    delta_l2_ratio_tokens,
    kl_divergence_tokens,
    logits_entropy_and_margin,
)
from src.gating.rule_gate import RuleGate, RuleGateConfig, StepGateResult
from src.gating.learned_gate import LearnedGate, LearnedGateConfig, LearnedGateWeights
from src.gating.conformal import (
    ConformalRiskCalibrator,
    StepwiseConformalCalibrator,
    compute_risk_score,
)
from src.gating.budget import BudgetController
from src.gating.adaptive import AdaptiveScheduler, AdaptiveSchedulerConfig
from src.gating.rollback import RollbackBuffer
from src.eval.tasks import load_wikitext2, load_lambada_openai, load_gsm8k_tiny
from src.utils.profile import query_gpu_utilization, GPUUtilSampler
from src.utils.quantiles import (
    DEFAULT_QUANTILES,
    StepQuantileTable,
    apply_ema,
    clip_sequence,
    compute_step_metric_quantiles,
    dump_step_quantiles,
)
from src.utils.config import load_yaml, ensure_output_dirs, save_manifest

try:  # Optional dependency; scripts can run without W&B
    import wandb  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    wandb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delta-Compute dLLM PoC runner")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["teacher", "rule_gate", "learned_gate", "adaptive", "oracle", "baselines"],
        required=True,
    )
    parser.add_argument("--engine", type=str, default=None, choices=["mock", "d2f"], help="Engine backend")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for runs")
    parser.add_argument("--num_prompts", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--consistency_check", action="store_true", help="Run full-compute baseline to check final token consistency")
    parser.add_argument("--config", type=str, default=None, help="YAML config with paths and params")
    parser.add_argument("--phase", type=str, default=None, help="Experiment phase: P1|P2|P3")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--models_root", type=str, default=None)
    parser.add_argument("--outputs_root", type=str, default=None)
    parser.add_argument("--use_d2f_lora", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--task", type=str, default=None, choices=["synthetic", "wikitext", "lambada", "gsm8k"], help="Prompt source")
    parser.add_argument("--tasks_cfg", type=str, default=None, help="YAML config for tasks")
    parser.add_argument("--paths_cfg", type=str, default=None, help="YAML config for data/model/output paths")
    parser.add_argument("--thresholds_cfg", type=str, default=None, help="YAML config for gating thresholds")
    parser.add_argument("--profile", type=str, default=None, help="Threshold profile name (balanced/aggressive/...)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset key within thresholds config")
    parser.add_argument("--dump_features_dir", type=str, default=None, help="Directory to dump token-level features for training the learned gate")
    parser.add_argument("--label_cos_tau", type=float, default=None, help="Cosine threshold for positive labels when dumping features")
    parser.add_argument("--label_dl2_rho", type=float, default=None, help="ΔL2 threshold for positive labels when dumping features")
    parser.add_argument("--label_kl_max", type=float, default=None, help="KL threshold for positive labels when dumping features")
    parser.add_argument("--learned_gate_weights", type=str, default=None, help="Path to learned gate weight file (.npz)")
    parser.add_argument("--learned_gate_threshold", type=float, default=None, help="Probability threshold for the learned gate")
    parser.add_argument("--learned_gate_freeze_K", type=int, default=None, help="Freeze horizon for learned gate")
    parser.add_argument("--learned_gate_min_consecutive", type=int, default=None, help="Consecutive stable steps before freezing in learned gate")
    # Rule gate thresholds
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--freeze_K", type=int, default=None)
    parser.add_argument("--watchdog_min_cos", type=float, default=None)
    parser.add_argument("--watchdog_max_kl", type=float, default=None)
    parser.add_argument("--layer_recompute_M", type=int, default=None)
    parser.add_argument("--gamma_margin", type=float, default=None, help="Margin threshold override for gating decisions")
    parser.add_argument("--quantiles_path", type=str, default=None, help="Path to teacher stepwise quantiles JSON")
    parser.add_argument("--one_minus_cos_quantile", type=float, default=None, help="Quantile level (0-1) used to derive per-step cosine thresholds")
    parser.add_argument("--delta_l2_quantile", type=float, default=None, help="Quantile level (0-1) used to derive per-step ΔL2 thresholds")
    parser.add_argument("--kl_quantile", type=float, default=None, help="Quantile level (0-1) used to derive per-step KL guard thresholds")
    parser.add_argument("--stepwise_ema", type=float, default=None, help="EMA smoothing factor (0-1, high=more smoothing) for stepwise thresholds")
    parser.add_argument("--stepwise_clip", type=str, default=None, help="Comma-separated min,max bounds applied to raw stepwise thresholds")
    parser.add_argument("--risk_quantiles_path", type=str, default=None, help="Path to precomputed per-step risk thresholds for conformal calibration")
    parser.add_argument("--risk_initial_quantile", type=float, default=None, help="Fallback quantile used before enough unsafe samples exist")
    parser.add_argument("--risk_low_support", type=int, default=None, help="Minimum unsafe sample count per step before using per-step threshold")
    parser.add_argument("--risk_window", type=int, default=None, help="Sliding window size for stepwise conformal calibrator")
    parser.add_argument("--consistency_full_compute", action="store_true", help="Run full-compute baseline and compare outputs; write difference report")
    parser.add_argument("--risk_delta", type=float, default=0.01, help="Target conformal δ bound")
    parser.add_argument("--budget_fraction", type=float, default=1.0, help="Per-step compute budget as fraction of Teacher FLOPs")
    parser.add_argument("--oracle_eval", action="store_true", help="When running Teacher, also compute oracle headroom metrics")
    parser.add_argument("--lte_probe", type=str, default="euler", choices=["none", "euler", "heun"], help="LTE probing strategy")
    parser.add_argument("--lte_eps", type=float, default=0.2, help="LTE tolerance for adaptive scheduler (interpreted in raw ΔL2 units by default)")
    parser.add_argument("--lte_min_consec", type=int, default=2, help="Consecutive LTE acceptances before increasing stride")
    parser.add_argument("--max_stride", type=int, default=4, help="Maximum adaptive stride")
    parser.add_argument("--adaptive_budget", type=float, default=0.20, help="Adaptive total skip budget")
    parser.add_argument("--sprt_alpha", type=float, default=0.005, help="SPRT Type I error")
    parser.add_argument("--sprt_beta", type=float, default=0.005, help="SPRT Type II error")
    parser.add_argument("--adaptive_hard_budget", action="store_true", help="Enforce skip_budget as a hard cap in adaptive mode")
    return parser.parse_args()


def coalesce(*values):
    for value in values:
        if value is not None:
            return value
    return None


def load_default_paths(custom_path: Optional[str] = None) -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    candidate = Path(custom_path) if custom_path else (repo_root / "configs" / "paths.yaml")
    if candidate.exists():
        return load_yaml(str(candidate))
    return {}


def synthetic_prompts(num_prompts: int) -> List[str]:
    base_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A wizard's job is to vex chumps quickly in fog.",
        "Pack my box with five dozen liquor jugs.",
        "Sphinx of black quartz, judge my vow.",
        "How vexingly quick daft zebras jump!",
    ]
    prompts: List[str] = []
    for i in range(num_prompts):
        prompts.append(base_sentences[i % len(base_sentences)])
    return prompts


def format_gsm8k_prompt_cot(question: str) -> str:
    """0-shot CoT style prompt for GSM8K.

    We ask the model to think step by step, then produce the final numeric answer
    strictly in the form '#### <number>' on a new last line. This matches our
    answer extraction and Dream's recommended eval style.
    """
    instruction = (
        "You are a helpful math assistant. Solve the problem step by step. "
        "After the reasoning, output the final answer on a new last line in the exact format '#### <number>' without any extra text."
    )
    return (
        f"{instruction}\n\n"
        f"Q: {question}\n"
        f"A: Let's think step by step."
    )


def load_prompts(num_prompts: int, task: str, data_root: Optional[str]) -> Tuple[List[str], Optional[List[str]]]:
    if task == "synthetic":
        return synthetic_prompts(num_prompts), None
    if task == "wikitext":
        return load_wikitext2(n_eval=num_prompts, data_root=data_root), None
    if task == "lambada":
        batch = load_lambada_openai(n_eval=num_prompts, data_root=data_root)
        return batch.texts, batch.labels
    if task == "gsm8k":
        batch = load_gsm8k_tiny(n_eval=num_prompts, data_root=data_root)
        formatted = [format_gsm8k_prompt_cot(q) for q in batch.texts]
        return formatted, batch.labels
    raise ValueError(f"Unsupported task: {task}")


def ensure_dirs(out_dir: str) -> Dict[str, str]:
    figures_dir = os.path.join(out_dir, "figures")
    runs_dir = os.path.join(out_dir, "runs")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    return {"figures": figures_dir, "runs": runs_dir}


def save_jsonl(path: str, rows: List[Dict]) -> None:
    # Ensure parent directory exists to avoid race/cleanup issues
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


FEATURE_NAMES = (
    "cosine",
    "delta_l2",
    "kl",
    "entropy",
    "margin",
    "step_frac",
    "sigma",
    "cosine_norm",
    "delta_l2_norm",
    "risk_score",
)


_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def slugify(value: Optional[str], fallback: str = "run") -> str:
    if not value:
        return fallback
    candidate = str(value).strip().replace(" ", "_")
    candidate = _SLUG_RE.sub("-", candidate)
    candidate = candidate.strip("-_")
    return candidate or fallback


def parse_clip_config(value: Optional[object]) -> Tuple[Optional[float], Optional[float]]:
    if value is None:
        return (None, None)
    if isinstance(value, str):
        parts = [part for part in value.split(",") if part]
    elif isinstance(value, (list, tuple)):
        parts = [str(part) for part in value]
    else:
        raise ValueError("stepwise_clip must be a comma-separated string or length-2 sequence")
    if not parts:
        return (None, None)
    if len(parts) == 1:
        lo = float(parts[0])
        return (lo, None)
    lo = float(parts[0]) if parts[0] else None
    hi = float(parts[1]) if parts[1] else None
    return (lo, hi)


@dataclass
class TeacherArtifacts:
    features_path: Optional[str] = None
    quantiles_path: Optional[str] = None
    outputs_path: Optional[str] = None


@dataclass
class StepwiseThresholds:
    cosine_tau: List[float]
    delta_l2_rho: List[float]
    gamma_margin: Optional[List[float]] = None
    source_path: Optional[str] = None

    def cosine_for_step(self, step: int) -> Optional[float]:
        if 0 <= step < len(self.cosine_tau):
            return float(self.cosine_tau[step])
        return None

    def delta_for_step(self, step: int) -> Optional[float]:
        if 0 <= step < len(self.delta_l2_rho):
            return float(self.delta_l2_rho[step])
        return None

    def margin_for_step(self, step: int) -> Optional[float]:
        if self.gamma_margin is None:
            return None
        if 0 <= step < len(self.gamma_margin):
            return float(self.gamma_margin[step])
        return None


def logits_argmax_and_margin(logits: np.ndarray) -> Tuple[List[int], List[float]]:
    top_indices = np.argmax(logits, axis=-1)
    sorted_logits = np.sort(logits, axis=-1)
    top1_vals = sorted_logits[:, -1]
    if logits.shape[1] > 1:
        top2_vals = sorted_logits[:, -2]
    else:
        top2_vals = np.zeros_like(top1_vals)
    margins = top1_vals - top2_vals
    return top_indices.astype(np.int64).tolist(), margins.astype(np.float32).tolist()


def _accumulate_ppl(stats: Dict[str, float], logits: np.ndarray, input_ids: Sequence[int], pad_token_id: int, special_mask: Optional[Sequence[int]]) -> None:
    if logits is None or input_ids is None:
        return
    ids = np.asarray(input_ids, dtype=np.int64)
    if ids.ndim == 0:
        ids = ids.reshape(1)
    if logits.shape[0] <= 1 or ids.shape[0] <= 1:
        return

    ids_next = ids[1:]
    logits_used = logits[:-1]
    mask = np.ones_like(ids_next, dtype=bool)

    if pad_token_id is not None and pad_token_id != -1:
        mask &= ids_next != int(pad_token_id)
    if special_mask is not None:
        spec = np.asarray(special_mask, dtype=bool)
        if spec.shape == ids.shape:
            mask &= ~spec[1:]

    valid_positions = np.nonzero(mask)[0]
    if valid_positions.size == 0:
        return

    target_ids = ids_next[valid_positions].astype(np.int64)
    logits_valid = logits_used[valid_positions].astype(np.float64)
    if logits_valid.ndim != 2 or logits_valid.shape[0] == 0:
        return

    logits_centered = logits_valid - np.max(logits_valid, axis=-1, keepdims=True)
    log_probs = logits_centered - np.log(np.sum(np.exp(logits_centered), axis=-1, keepdims=True))
    token_nll = -log_probs[np.arange(target_ids.shape[0]), target_ids]

    stats["nll_sum"] = stats.get("nll_sum", 0.0) + float(np.sum(token_nll))
    stats["token_count"] = stats.get("token_count", 0.0) + float(token_nll.shape[0])


def _normalize_word(word: str) -> str:
    return word.strip().strip("\"'`.,;:!?()[]{}<>-_\u201c\u201d\u2018\u2019").lower()


def _extract_last_word(text: str) -> str:
    tokens = text.strip().split()
    return tokens[-1] if tokens else ""


def _decode_last_token_text(logits: Optional[np.ndarray], engine: BaseEngine) -> str:
    if logits is None or logits.ndim == 0 or logits.shape[0] == 0:
        return ""
    last_row = logits[-1]
    if last_row.ndim == 0 or last_row.size == 0:
        return ""
    token_id = int(np.argmax(last_row))
    try:
        return engine.decode_tokens([token_id]).strip()
    except NotImplementedError:
        return ""


def _generate_text(engine: BaseEngine, prompt: str, max_new: int) -> str:
    try:
        ids = engine.greedy_generate(prompt, max_new_tokens=max_new)
        return engine.decode_tokens(ids).strip()
    except NotImplementedError:
        return ""
    except Exception:
        # Generation not supported by the current engine/model; degrade gracefully
        return ""


def _safe_decode(engine: BaseEngine, token_ids: Sequence[int]) -> str:
    try:
        return engine.decode_tokens(token_ids).strip()
    except Exception:
        return ""


def _dream_completion_and_prompt(
    engine: BaseEngine,
    base_prompt: str,
    max_new: int,
) -> Tuple[str, str]:
    """Generate a completion for GSM8K-style prompts and return the completion text
    along with the augmented prompt (base prompt + completion)."""
    gen_prompt = _maybe_chat_wrap(engine, base_prompt)
    completion_ids: Sequence[int] = []
    completion_text = ""
    try:
        completion_ids = engine.greedy_generate(gen_prompt, max_new_tokens=max_new)
        completion_text = engine.decode_tokens(completion_ids).strip()
    except Exception:
        completion_ids = []
        completion_text = ""
    if completion_text:
        augmented_prompt = f"{base_prompt.rstrip()}\n\n{completion_text}"
        return completion_text, augmented_prompt
    return "", base_prompt


def _evaluation_text_from_diffusion(
    diffusion_text: Optional[str],
    prompt: str,
    engine: BaseEngine,
    max_new: int,
) -> Tuple[str, str]:
    """
    Returns the text used for scoring along with its source.
    Source is 'diffusion' when the decoded diffusion text is non-empty,
    otherwise 'greedy_fallback' after regenerating from the prompt.
    """
    if diffusion_text and diffusion_text.strip():
        return diffusion_text.strip(), "diffusion"
    regen_prompt = _maybe_chat_wrap(engine, prompt)
    regenerated = _generate_text(engine, regen_prompt, max_new=max_new)
    return regenerated.strip(), "greedy_fallback"

def _maybe_chat_wrap(engine: BaseEngine, prompt: str) -> str:
    """If the engine supports chat templates (e.g., Dream Instruct), wrap the
    raw prompt into a single-user chat turn so generation follows the
    expected format. Otherwise return the prompt unchanged.
    """
    try:
        # type: ignore[attr-defined]
        apply_chat = getattr(engine, "apply_chat_template", None)
        if callable(apply_chat):
            return apply_chat([{"role": "user", "content": prompt}], add_generation_prompt=True)
    except Exception:
        pass
    return prompt


def _extract_gsm_answer(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if "####" in text:
        return text.split("####")[-1].strip()
    matches = re.findall(r"-?\d[\d,\.]*", text)
    if matches:
        return matches[-1]
    tokens = text.split()
    return tokens[-1] if tokens else text


def _normalize_gsm_answer(ans: str) -> str:
    return ans.strip().replace(",", "").replace(" ", "").lower()


@dataclass
class TokenObservables:
    cosine: np.ndarray
    delta_l2: np.ndarray
    kl: np.ndarray
    entropy: np.ndarray
    margin: np.ndarray
    step_frac: np.ndarray
    sigma: np.ndarray
    cosine_norm: np.ndarray
    delta_l2_norm: np.ndarray


def sigma_for_step(aux: Optional[Dict], step_index: int, num_steps: int) -> float:
    if aux and "sigma" in aux:
        try:
            return float(aux["sigma"])
        except Exception:
            pass
    denom = max(1, num_steps - 1)
    # Simple cosine schedule as placeholder when the engine does not report σ_t
    ratio = float(step_index) / float(denom)
    return float(max(1e-3, np.cos(ratio * np.pi / 2.0)))


def compute_token_observables(
    current_hidden: List[np.ndarray],
    prev_hidden: List[np.ndarray],
    current_logits: np.ndarray,
    prev_logits: np.ndarray,
    step_index: int,
    num_steps: int,
    aux: Optional[Dict] = None,
    valid_mask: Optional[np.ndarray] = None,
) -> TokenObservables:
    cos = cosine_similarity_tokens(current_hidden, prev_hidden)
    dl2 = delta_l2_ratio_tokens(current_hidden, prev_hidden)
    kl = kl_divergence_tokens(current_logits, prev_logits)
    entropy, margin = logits_entropy_and_margin(current_logits)
    if valid_mask is not None:
        cos = cos[valid_mask]
        dl2 = dl2[valid_mask]
        kl = kl[valid_mask]
        entropy = entropy[valid_mask]
        margin = margin[valid_mask]
    denom = max(1, num_steps - 1)
    step_frac = np.full_like(cos, fill_value=float(step_index) / float(denom), dtype=np.float32)
    sigma = float(sigma_for_step(aux, step_index, num_steps))
    sigma_arr = np.full_like(cos, fill_value=sigma, dtype=np.float32)
    sigma_safe = np.maximum(sigma_arr, 1e-3)
    cos_norm = np.clip((cos + 1.0) / 2.0, 0.0, 1.0)
    dl2_norm = dl2 / sigma_safe
    return TokenObservables(
        cosine=cos,
        delta_l2=dl2,
        kl=kl,
        entropy=entropy,
        margin=margin,
        step_frac=step_frac,
        sigma=sigma_arr,
        cosine_norm=cos_norm,
        delta_l2_norm=dl2_norm,
    )


def run_teacher(
    engine: BaseEngine,
    prompts: List[str],
    out_dirs: Dict[str, str],
    num_steps: int,
    dump_features_dir: Optional[str] = None,
    label_thresholds: Optional[Dict[str, float]] = None,
    oracle_eval: bool = False,
    *,
    task_name: Optional[str] = None,
    exp_name: Optional[str] = None,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
    labels: Optional[List[str]] = None,
    gen_max_new: int = 128,
) -> TeacherArtifacts:
    from src.viz.plots import save_heatmap, save_hist, save_errorbar

    aggregates: List[Dict] = []
    cos_means: List[List[float]] = []
    dl2_means: List[List[float]] = []
    kl_means: List[List[float]] = []
    ent_means: List[List[float]] = []
    dl2_norm_means: List[List[float]] = []
    mse_series: List[List[float]] = []
    cos_hist_samples: List[float] = []
    dl2_hist_samples: List[float] = []
    dl2_norm_hist_samples: List[float] = []
    stability_lengths: List[int] = []

    feature_blocks: List[np.ndarray] = []
    feature_labels: List[np.ndarray] = []
    feature_meta: List[np.ndarray] = []
    oracle_ratios: List[float] = []

    per_seq_latency_ms: List[float] = []
    total_start = time.time()
    sampler = GPUUtilSampler(interval_sec=0.5)
    sampler.start()

    label_cfg = label_thresholds or {}
    cos_tau = float(label_cfg.get("cosine_tau", 0.97))
    dl2_rho = float(label_cfg.get("delta_l2_rho", 0.05))
    kl_max = float(label_cfg.get("kl_max", 0.01))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    total_time_s = max(1e-6, time.time() - total_start)
    layer_step_mse: Dict[Tuple[int, int], List[float]] = defaultdict(list)

    metric_samples = {
        "one_minus_cos": defaultdict(list),
        "delta_l2_norm": defaultdict(list),
        "kl": defaultdict(list),
    }
    step_token_counts: Dict[int, int] = defaultdict(int)
    features_path: Optional[str] = None
    teacher_outputs_path = os.path.join(out_dirs["runs"], f"teacher_outputs_{timestamp}.jsonl")
    teacher_outputs_file = open(teacher_outputs_path, "w", encoding="utf-8")

    is_wikitext = task_name == "wikitext"
    is_lambada = task_name == "lambada"
    is_gsm8k = task_name == "gsm8k"
    ppl_stats: Dict[str, float] = {"nll_sum": 0.0, "token_count": 0.0}
    lambada_correct = 0
    lambada_total = len(labels) if is_lambada and labels is not None else 0
    gsm_correct = 0
    gsm_total = len(labels) if is_gsm8k and labels is not None else 0
    first_prompt_step_trace: List[Dict[str, Any]] = []
    layer_prompt_cos: Dict[int, List[float]] = defaultdict(list)
    layer_cos_means: Dict[int, List[float]] = defaultdict(list)
    layer_cos_stds: Dict[int, List[float]] = defaultdict(list)
    layer_dl2_means: Dict[int, List[float]] = defaultdict(list)
    layer_dl2_stds: Dict[int, List[float]] = defaultdict(list)

    for prompt_idx, prompt in enumerate(prompts):
        base_prompt = prompt
        dream_completion: Optional[str] = None
        if is_gsm8k:
            completion_text, augmented_prompt = _dream_completion_and_prompt(engine, base_prompt, gen_max_new)
            if completion_text:
                dream_completion = completion_text
                prompt = augmented_prompt
        prompt_for_engine = prompt
        state = engine.encode_prompt(prompt)
        prev_hidden: Optional[List[np.ndarray]] = None
        prev_logits: Optional[np.ndarray] = None
        prev_aux: Optional[Dict] = None
        step_cos_means: List[float] = []
        step_dl2_means: List[float] = []
        step_kl_means: List[float] = []
        step_ent_means: List[float] = []
        step_dl2_norm_means: List[float] = []
        prompt_mse: List[float] = []
        stability_counter: Optional[np.ndarray] = None
        stability_max: Optional[np.ndarray] = None
        # Per-layer accumulation for mean and IQR
        layer_step_values_cos: Dict[Tuple[int, int], np.ndarray] = {}
        layer_step_values_dl2: Dict[Tuple[int, int], np.ndarray] = {}

        seq_start = time.time()
        final_aux: Optional[Dict] = None
        capture_trace = prompt_idx == 0
        prompt_step_trace: List[Dict[str, Any]] = []
        for t in range(num_steps):
            out = engine.step(state, t)
            # Immediate GPU sample after a GPU-heavy op to avoid missing short spikes
            try:
                sampler.sample_once()
            except Exception:
                pass
            final_aux = out.aux if isinstance(out.aux, dict) else None
            step_tokens, _ = logits_argmax_and_margin(out.logits)
            try:
                step_decoded = engine.decode_tokens(step_tokens).strip()
            except Exception:
                step_decoded = ""
            oracle_skip_fraction: Optional[float] = None
            oracle_skip_indices: Optional[List[int]] = None
            if prev_hidden is not None and prev_logits is not None:
                # Build valid mask (exclude padding and special tokens)
                valid_mask: Optional[np.ndarray] = None
                if isinstance(out.aux, dict):
                    ids = out.aux.get("input_ids")
                    special = out.aux.get("special_tokens_mask")
                    pad_id = out.aux.get("pad_token_id", -1)
                    if ids is not None:
                        ids = np.array(ids, dtype=np.int64)
                        is_pad = (ids == pad_id) if pad_id != -1 else np.zeros_like(ids, dtype=bool)
                        is_special = np.array(special, dtype=bool) if special is not None else np.zeros_like(ids, dtype=bool)
                        valid_mask = ~(is_pad | is_special)
                obs = compute_token_observables(
                    out.hidden_by_layer,
                    prev_hidden,
                    out.logits,
                    prev_logits,
                    t,
                    num_steps,
                    aux=out.aux if isinstance(out.aux, dict) else None,
                    valid_mask=valid_mask,
                )
                cos = obs.cosine
                dl2 = obs.delta_l2
                kl = obs.kl
                ent = obs.entropy
                margin = obs.margin
                step_frac = obs.step_frac
                dl2_norm = obs.delta_l2_norm
                # Record oracle skip ratio independent of feature dumping
                oracle_mask = ((cos >= cos_tau) & (dl2 <= dl2_rho) & (kl <= kl_max)).astype(np.float32)
                oracle_ratios.append(float(np.mean(oracle_mask)))
                step_cos_means.append(float(np.mean(cos)))
                step_dl2_means.append(float(np.mean(dl2)))
                step_kl_means.append(float(np.mean(kl)))
                step_ent_means.append(float(np.mean(ent)))
                step_dl2_norm_means.append(float(np.mean(dl2_norm)))
                cos_hist_samples.extend(cos.flatten().tolist())
                dl2_hist_samples.extend(dl2.flatten().tolist())
                dl2_norm_hist_samples.extend(dl2_norm.flatten().tolist())
                metric_samples["one_minus_cos"][t].append(1.0 - cos)
                metric_samples["delta_l2_norm"][t].append(dl2_norm)
                metric_samples["kl"][t].append(kl)
                step_token_counts[t] += int(cos.shape[0])
                if stability_counter is None or stability_counter.shape[0] != oracle_mask.shape[0]:
                    stability_counter = np.zeros_like(oracle_mask, dtype=np.int32)
                    stability_max = np.zeros_like(oracle_mask, dtype=np.int32)
                stable_mask = oracle_mask.astype(bool)
                stability_counter[stable_mask] += 1
                stability_counter[~stable_mask] = 0
                if stability_max is not None:
                    np.maximum(stability_max, stability_counter, out=stability_max)
                layer_mse_values: List[float] = []
                for h_now, h_prev in zip(out.hidden_by_layer, prev_hidden):
                    diff = h_now - h_prev
                    layer_mse_values.append(float(np.mean(np.square(diff))))
                if layer_mse_values:
                    prompt_mse.append(float(np.mean(layer_mse_values)))
                    for li, mse_value in enumerate(layer_mse_values):
                        layer_step_mse[(li, t)].append(mse_value)
                # Save per-layer token vectors for IQR
                num_layers_current = len(out.hidden_by_layer)
                for li, (h_now, h_prev) in enumerate(zip(out.hidden_by_layer, prev_hidden)):
                    # Compute per-token layer-wise metrics (masked)
                    cos_layer = cosine_similarity_tokens([h_now], [h_prev])
                    dl2_layer = delta_l2_ratio_tokens([h_now], [h_prev])
                    if valid_mask is not None:
                        cos_layer = cos_layer[valid_mask]
                        dl2_layer = dl2_layer[valid_mask]
                    layer_step_values_cos[(li, t)] = cos_layer
                    layer_step_values_dl2[(li, t)] = dl2_layer
                    if t == num_steps - 1:
                        mean_cos_layer = float(np.mean(cos_layer)) if cos_layer.size else float("nan")
                        layer_prompt_cos[li].append(mean_cos_layer)
                # record per-layer stats for plots
                for li in range(num_layers_current):
                    cos_vals = layer_step_values_cos.get((li, t))
                    dl2_vals = layer_step_values_dl2.get((li, t))
                    if cos_vals is not None and cos_vals.size:
                        layer_cos_means[li].append(float(np.mean(cos_vals)))
                        layer_cos_stds[li].append(float(np.std(cos_vals)))
                    else:
                        layer_cos_means[li].append(float("nan"))
                        layer_cos_stds[li].append(float("nan"))
                    if dl2_vals is not None and dl2_vals.size:
                        layer_dl2_means[li].append(float(np.mean(dl2_vals)))
                        layer_dl2_stds[li].append(float(np.std(dl2_vals)))
                    else:
                        layer_dl2_means[li].append(float("nan"))
                        layer_dl2_stds[li].append(float("nan"))
                oracle_mask_bool = oracle_mask.astype(bool)
                oracle_skip_fraction = float(np.mean(oracle_mask_bool))
                oracle_skip_indices = [int(i) for i, flag in enumerate(oracle_mask_bool) if flag]
                aggregates.append(
                    {
                        "mode": "teacher",
                        "prompt_len": len(prompt),
                        "step": t,
                        "cos_mean": float(np.mean(cos)),
                        "dl2_mean": float(np.mean(dl2)),
                        "kl_mean": float(np.mean(kl)),
                        "entropy_mean": float(np.mean(ent)),
                        "margin_mean": float(np.mean(margin)),
                    }
                )

                if dump_features_dir:
                    risk_teacher = compute_risk_score(obs.cosine_norm, obs.delta_l2_norm, obs.kl)
                    features = np.stack(
                        [
                            cos,
                            dl2,
                            kl,
                            ent,
                            margin,
                            step_frac,
                            obs.sigma,
                            obs.cosine_norm,
                            obs.delta_l2_norm,
                            risk_teacher,
                        ],
                        axis=-1,
                    ).astype(np.float32)
                    feature_blocks.append(features.reshape(-1, features.shape[-1]))
                    feature_labels.append(oracle_mask.reshape(-1))
                    shape = oracle_mask.reshape(-1).shape
                    meta = np.stack(
                        [
                            np.full(shape, prompt_idx, dtype=np.int32),
                            np.full(shape, t, dtype=np.int32),
                        ],
                        axis=-1,
                    )
                    feature_meta.append(meta)
            # Snapshot prev strictly after metrics (prevent view overwrite)
            prev_hidden = [h.copy() for h in out.hidden_by_layer]
            prev_logits = out.logits.copy()
            prev_aux = out.aux if isinstance(out.aux, dict) else None
            if capture_trace:
                prompt_step_trace.append(
                    {
                        "step": int(t),
                        "decoded_text": step_decoded,
                        "oracle_skip_fraction": oracle_skip_fraction,
                        "oracle_skip_indices": oracle_skip_indices,
                    }
                )

        cos_means.append(step_cos_means)
        dl2_means.append(step_dl2_means)
        kl_means.append(step_kl_means)
        ent_means.append(step_ent_means)
        dl2_norm_means.append(step_dl2_norm_means)
        mse_series.append(prompt_mse)
        if stability_max is not None:
            stability_lengths.extend(int(x) for x in stability_max.tolist())
        per_seq_latency_ms.append((time.time() - seq_start) * 1000.0)

        if prev_logits is not None:
            if is_wikitext and final_aux is not None:
                input_ids = final_aux.get("input_ids") if isinstance(final_aux, dict) else None
                special_mask = final_aux.get("special_tokens_mask") if isinstance(final_aux, dict) else None
                pad_token_id = int(final_aux.get("pad_token_id", -1)) if isinstance(final_aux, dict) else -1
                if input_ids is not None:
                    _accumulate_ppl(ppl_stats, prev_logits, input_ids, pad_token_id, special_mask)

            tokens, margins = logits_argmax_and_margin(prev_logits)
            checksum = float(np.sum(prev_logits)) if prev_logits.size else 0.0
            decoded_text = _safe_decode(engine, tokens)
            diffusion_text = decoded_text
            eval_text_val: Optional[str] = None
            eval_text_source: Optional[str] = None
            if dream_completion:
                eval_text_val = dream_completion
                eval_text_source = "dream_completion"
            elif (is_lambada or is_gsm8k) and labels is not None and prompt_idx < len(labels):
                eval_text_val, eval_text_source = _evaluation_text_from_diffusion(
                    diffusion_text, prompt, engine, gen_max_new
                )
            gen_text_lbd = None
            pred_word = None
            gen_text_gsm = None
            pred_ans = None
            if is_lambada and labels is not None and prompt_idx < len(labels):
                gold_word = labels[prompt_idx]
                gen_text_lbd = eval_text_val or ""
                pred_word = _extract_last_word(gen_text_lbd.split()[0] if gen_text_lbd else "")
                if _normalize_word(pred_word) == _normalize_word(str(gold_word)):
                    lambada_correct += 1
            if is_gsm8k and labels is not None and prompt_idx < len(labels):
                gold_ans = labels[prompt_idx]
                gen_text_gsm = eval_text_val or ""
                pred_ans = _normalize_gsm_answer(_extract_gsm_answer(gen_text_gsm))
                if pred_ans == _normalize_gsm_answer(str(gold_ans)):
                    gsm_correct += 1

            record = {
                "prompt_id": int(prompt_idx),
                "prompt": prompt,
                "tokens": tokens,
                "margins": margins,
                "logits_checksum": checksum,
            }
            record["diffusion_text"] = diffusion_text
            if is_lambada:
                record["gen_text_lambada"] = gen_text_lbd
                if eval_text_source is not None:
                    record["gen_text_lambada_source"] = eval_text_source
                record["pred_word"] = pred_word
            if is_gsm8k:
                record["gen_text_gsm8k"] = gen_text_gsm
                if eval_text_source is not None:
                    record["gen_text_gsm8k_source"] = eval_text_source
                record["pred_ans"] = pred_ans
                if dream_completion:
                    record["dream_completion_text"] = dream_completion
            teacher_outputs_file.write(json.dumps(record) + "\n")
        if capture_trace:
            first_prompt_step_trace = prompt_step_trace

    layer_prompt_arr: Optional[np.ndarray] = None
    if layer_prompt_cos:
        layer_ids = sorted(layer_prompt_cos.keys())
        num_prompts_total = len(prompts)
        arr = np.full((len(layer_ids), num_prompts_total), np.nan, dtype=np.float32)
        for idx, layer in enumerate(layer_ids):
            values = layer_prompt_cos[layer]
            if len(values) < num_prompts_total:
                values = values + [float("nan")] * (num_prompts_total - len(values))
            arr[idx, :len(values)] = np.asarray(values[:num_prompts_total], dtype=np.float32)
        layer_prompt_arr = arr

    layer_indices_order = sorted(layer_cos_means.keys())
    cosine_layer_profile: Optional[Tuple[np.ndarray, np.ndarray]] = None
    dl2_layer_profile: Optional[Tuple[np.ndarray, np.ndarray]] = None
    cosine_layer_err: Optional[np.ndarray] = None
    dl2_layer_err: Optional[np.ndarray] = None
    num_prompts_total = len(prompts)
    steps_recorded = max(0, num_steps - 1)
    if layer_indices_order and steps_recorded > 0 and num_prompts_total > 0:
        cos_means_vals: List[float] = []
        cos_err_vals: List[float] = []
        dl2_means_vals: List[float] = []
        dl2_err_vals: List[float] = []
        valid_layers: List[int] = []
        for li in layer_indices_order:
            cos_series = np.asarray(layer_cos_means[li], dtype=np.float32)
            dl2_series = np.asarray(layer_dl2_means[li], dtype=np.float32)
            if cos_series.size == 0:
                continue
            try:
                cos_matrix = cos_series.reshape(num_prompts_total, steps_recorded)
            except ValueError:
                cos_matrix = cos_series.reshape(-1)
            valid_layers.append(li)
            cos_flat = cos_matrix.flatten()
            cos_means_vals.append(float(np.nanmean(cos_flat)))
            cos_err_vals.append(float(np.nanstd(cos_flat)))
            if dl2_series.size:
                try:
                    dl2_matrix = dl2_series.reshape(num_prompts_total, steps_recorded)
                except ValueError:
                    dl2_matrix = dl2_series.reshape(-1)
                dl2_flat = dl2_matrix.flatten()
                dl2_means_vals.append(float(np.nanmean(dl2_flat)))
                dl2_err_vals.append(float(np.nanstd(dl2_flat)))
            else:
                dl2_means_vals.append(float("nan"))
                dl2_err_vals.append(float("nan"))
        if cos_means_vals:
            cosine_layer_profile = (
                np.asarray(valid_layers, dtype=np.int32),
                np.asarray(cos_means_vals, dtype=np.float32),
            )
            cosine_layer_err = np.asarray(cos_err_vals, dtype=np.float32)
        else:
            cosine_layer_profile = None
            cosine_layer_err = None
        if dl2_means_vals:
            dl2_layer_profile = (
                np.asarray(valid_layers, dtype=np.int32),
                np.asarray(dl2_means_vals, dtype=np.float32),
            )
            dl2_layer_err = np.asarray(dl2_err_vals, dtype=np.float32)
        else:
            dl2_layer_profile = None
            dl2_layer_err = None
    else:
        cosine_layer_err = None
        dl2_layer_err = None

    layer_mse_curves: Dict[int, List[float]] = {}
    mse_steps: List[int] = list(range(1, num_steps)) if num_steps > 1 else []
    if layer_step_mse and mse_steps:
        layer_indices = sorted(set(li for li, _ in layer_step_mse.keys()))
        for li in layer_indices:
            series: List[float] = []
            for step in mse_steps:
                values = layer_step_mse.get((li, step))
                if values:
                    series.append(float(np.mean(values)))
                else:
                    series.append(float("nan"))
            layer_mse_curves[li] = series

    save_jsonl(os.path.join(out_dirs["runs"], f"teacher_{timestamp}.jsonl"), aggregates)
    # Export layer×step mean and IQR to CSV
    teacher_csv = os.path.join(out_dirs["runs"], f"teacher_aggregates_{timestamp}.csv")
    with open(teacher_csv, "w", encoding="utf-8") as f:
        f.write("layer,step,metric,mean,q25,q75\n")
        # infer num_layers, num_steps from collected keys
        cos_keys = list(layer_step_values_cos.keys())
        steps = sorted(set(k[1] for k in cos_keys))
        layers = sorted(set(k[0] for k in cos_keys))
        for li in layers:
            for t in steps:
                if (li, t) in layer_step_values_cos:
                    v = layer_step_values_cos[(li, t)]
                    f.write(f"{li},{t},cos,{np.mean(v):.6f},{np.percentile(v,25):.6f},{np.percentile(v,75):.6f}\n")
                if (li, t) in layer_step_values_dl2:
                    v = layer_step_values_dl2[(li, t)]
                    f.write(f"{li},{t},dl2,{np.mean(v):.6f},{np.percentile(v,25):.6f},{np.percentile(v,75):.6f}\n")

    if len(cos_means) > 0 and len(cos_means[0]) > 0:
        arr = np.array(cos_means).T  # steps x prompts
        save_heatmap(arr, os.path.join(out_dirs["figures"], f"teacher_cosine_heatmap_{timestamp}.png"), title="Cosine mean per step (prompts as columns)")
    if len(dl2_means) > 0 and len(dl2_means[0]) > 0:
        arr = np.array(dl2_means).T
        save_heatmap(arr, os.path.join(out_dirs["figures"], f"teacher_dl2_heatmap_{timestamp}.png"), title="ΔL2 mean per step")
    if layer_prompt_arr is not None and layer_prompt_arr.size:
        save_heatmap(layer_prompt_arr, os.path.join(out_dirs["figures"], f"teacher_layer_prompt_heatmap_{timestamp}.png"), title="Final-step cosine mean per layer", xlabel="prompts", ylabel="layers")
    if cosine_layer_profile is not None and cosine_layer_err is not None:
        cos_err = np.nan_to_num(cosine_layer_err, nan=0.0)
        save_errorbar(
            cosine_layer_profile[0],
            cosine_layer_profile[1],
            cos_err,
            os.path.join(out_dirs["figures"], f"teacher_layer_cosine_profile_{timestamp}.png"),
            title="Cosine similarity between consecutive diffusion steps",
            xlabel="layer index",
            ylabel="cosine similarity",
            color="crimson",
        )
    if dl2_layer_profile is not None and dl2_layer_err is not None:
        dl2_err = np.nan_to_num(dl2_layer_err, nan=0.0)
        save_errorbar(
            dl2_layer_profile[0],
            dl2_layer_profile[1],
            dl2_err,
            os.path.join(out_dirs["figures"], f"teacher_layer_dl2_profile_{timestamp}.png"),
            title="L2 norm ratio between consecutive diffusion steps",
            xlabel="layer index",
            ylabel="ΔL2 ratio",
            color="darkorange",
        )
    # Export KL/entropy step curves to CSV for further plotting
    if kl_means:
        kl_csv = os.path.join(out_dirs["runs"], f"teacher_kl_entropy_{timestamp}.csv")
        steps = list(range(len(kl_means[0]))) if kl_means[0] else []
        with open(kl_csv, "w", encoding="utf-8") as f:
            f.write("step,kl_mean,entropy_mean\n")
            for s in steps:
                km = float(np.mean([row[s] for row in kl_means if len(row) > s]))
                em = float(np.mean([row[s] for row in ent_means if len(row) > s]))
                f.write(f"{s},{km:.6f},{em:.6f}\n")
    # Export layer bands (L1-3, L4-6, L7+) IQR/mean
    if layer_step_values_cos:
        bands_csv = os.path.join(out_dirs["runs"], f"teacher_layer_bands_{timestamp}.csv")
        with open(bands_csv, "w", encoding="utf-8") as f:
            f.write("band,step,metric,mean,q25,q75\n")
            cos_keys = list(layer_step_values_cos.keys())
            steps = sorted(set(k[1] for k in cos_keys))
            layers = sorted(set(k[0] for k in cos_keys))
            # Define bands
            def band_of(li: int) -> str:
                if li <= 2:
                    return "L1-3"
                if li <= 5:
                    return "L4-6"
                return "L7+"
            for t in steps:
                band_to_vals_cos: Dict[str, List[float]] = {"L1-3": [], "L4-6": [], "L7+": []}
                band_to_vals_dl2: Dict[str, List[float]] = {"L1-3": [], "L4-6": [], "L7+": []}
                for li in layers:
                    b = band_of(li)
                    if (li, t) in layer_step_values_cos:
                        band_to_vals_cos[b].extend(layer_step_values_cos[(li, t)].tolist())
                    if (li, t) in layer_step_values_dl2:
                        band_to_vals_dl2[b].extend(layer_step_values_dl2[(li, t)].tolist())
                for b in ["L1-3", "L4-6", "L7+"]:
                    if band_to_vals_cos[b]:
                        v = np.array(band_to_vals_cos[b], dtype=np.float32)
                        f.write(f"{b},{t},cos,{np.mean(v):.6f},{np.percentile(v,25):.6f},{np.percentile(v,75):.6f}\n")
                    if band_to_vals_dl2[b]:
                        v = np.array(band_to_vals_dl2[b], dtype=np.float32)
                        f.write(f"{b},{t},dl2,{np.mean(v):.6f},{np.percentile(v,25):.6f},{np.percentile(v,75):.6f}\n")
    if len(cos_hist_samples) > 0:
        save_hist(np.array(cos_hist_samples), os.path.join(out_dirs["figures"], f"teacher_cosine_hist_{timestamp}.png"), title="Cosine similarity histogram")
    if len(dl2_hist_samples) > 0:
        save_hist(np.array(dl2_hist_samples), os.path.join(out_dirs["figures"], f"teacher_dl2_hist_{timestamp}.png"), title="ΔL2 histogram")
    if len(dl2_norm_hist_samples) > 0:
        save_hist(np.array(dl2_norm_hist_samples), os.path.join(out_dirs["figures"], f"teacher_dl2norm_hist_{timestamp}.png"), title="ΔL2 normalized histogram")
    if first_prompt_step_trace:
        trace_path = os.path.join(out_dirs["runs"], f"teacher_prompt0_trace_{timestamp}.jsonl")
        with open(trace_path, "w", encoding="utf-8") as trace_file:
            for entry in first_prompt_step_trace:
                trace_file.write(json.dumps(entry) + "\n")

    try:
        from src.viz.rich import render_teacher_suite

        render_teacher_suite(
            out_dirs["figures"],
            timestamp,
            cos_means,
            dl2_means,
            kl_means,
            ent_means,
            dl2_norm_means,
            cos_hist_samples,
            dl2_hist_samples,
            dl2_norm_hist_samples,
            stability_lengths,
            per_seq_latency_ms,
            mse_series,
            layer_mse_curves,
            mse_steps,
        )
    except Exception:
        pass

    total_time_s = max(1e-6, time.time() - total_start)
    sampler.stop()
    gpu = sampler.metrics or (query_gpu_utilization() or {})

    if dump_features_dir and feature_blocks:
        os.makedirs(dump_features_dir, exist_ok=True)
        features = np.concatenate(feature_blocks, axis=0)
        labels = np.concatenate(feature_labels, axis=0)
        meta = np.concatenate(feature_meta, axis=0)
        out_path = os.path.join(dump_features_dir, f"teacher_features_{timestamp}.npz")
        np.savez(
            out_path,
            features=features,
            labels=labels,
            meta=meta,
            feature_names=np.array(FEATURE_NAMES),
            label_thresholds=np.array([cos_tau, dl2_rho, kl_max], dtype=np.float32),
        )
        features_path = out_path

    quantiles_by_metric: Dict[str, Dict[int, Dict[str, float]]] = {}
    for metric_name, samples in metric_samples.items():
        quantiles_by_metric[metric_name] = compute_step_metric_quantiles(samples, quantile_levels)

    token_counts = {str(step): int(count) for step, count in step_token_counts.items()}
    dataset_slug = slugify(task_name, "dataset")
    exp_slug = slugify(exp_name, dataset_slug)
    quantile_filename = "_".join([
        "teacher_quantiles",
        dataset_slug,
        exp_slug,
        timestamp,
    ]) + ".json"
    quantiles_path = os.path.join(out_dirs["runs"], quantile_filename)
    dump_step_quantiles(
        quantiles_path,
        quantiles_by_metric,
        metadata={
            "task": task_name,
            "exp_name": exp_name,
            "num_steps": num_steps,
            "quantiles": list(float(q) for q in quantile_levels),
            "token_counts": token_counts,
            "timestamp": timestamp,
        },
    )

    quality_metrics: Dict[str, Dict[str, float]] = {}
    if is_wikitext and ppl_stats["token_count"] > 0:
        ppl_value = float(np.exp(ppl_stats["nll_sum"] / ppl_stats["token_count"]))
        quality_metrics["perplexity"] = {
            "value": ppl_value,
            "token_count": int(ppl_stats["token_count"]),
            "nll_sum": float(ppl_stats["nll_sum"]),
        }
    if is_lambada and lambada_total:
        quality_metrics["accuracy"] = {
            "value": float(lambada_correct / lambada_total if lambada_total else 0.0),
            "correct": int(lambada_correct),
            "total": int(lambada_total),
        }
    if is_gsm8k and gsm_total:
        quality_metrics["exact_match"] = {
            "value": float(gsm_correct / gsm_total if gsm_total else 0.0),
            "correct": int(gsm_correct),
            "total": int(gsm_total),
        }

    summary = {
        "mode": "teacher",
        "num_prompts": len(prompts),
        "num_steps": num_steps,
        "latency_ms_mean": float(np.mean(per_seq_latency_ms) if per_seq_latency_ms else 0.0),
        "latency_ms_p50": float(np.percentile(per_seq_latency_ms, 50) if per_seq_latency_ms else 0.0),
        "latency_ms_p90": float(np.percentile(per_seq_latency_ms, 90) if per_seq_latency_ms else 0.0),
        "throughput_seq_per_s": float(len(prompts) / total_time_s),
        "oracle_skip_ratio_mean": float(np.mean(oracle_ratios) if oracle_ratios else -1.0),
        "quantiles_path": quantiles_path,
        "features_path": features_path,
        "final_outputs_path": teacher_outputs_path,
        "task": task_name,
        "exp_name": exp_name,
        **{f"gpu_{k}": v for k, v in gpu.items()},
    }
    if quality_metrics:
        summary["quality"] = quality_metrics
    summary_path = os.path.join(out_dirs["runs"], f"teacher_summary_{timestamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    teacher_outputs_file.close()

    return TeacherArtifacts(features_path=features_path, quantiles_path=quantiles_path, outputs_path=teacher_outputs_path)


def run_rule_gate(
    engine: BaseEngine,
    prompts: List[str],
    out_dirs: Dict[str, str],
    num_steps: int,
    cfg: RuleGateConfig,
    risk_delta: float,
    budget_fraction: float,
    consistency_check: bool = False,
    stepwise_thresholds: Optional[StepwiseThresholds] = None,
    calibrator_cfg: Optional[Dict[str, object]] = None,
    decision_logger: Optional[DecisionLogger] = None,
    profile: Optional[str] = None,
    labels: Optional[List[str]] = None,
    task_name: Optional[str] = None,
    gen_max_new: int = 128,
) -> None:
    from src.viz.plots import save_hist

    aggregates: List[Dict] = []
    savings_ratios: List[float] = []
    freeze_ratios_by_step: Dict[int, List[float]] = defaultdict(list)

    gate = RuleGate(cfg)
    calibrator_params = calibrator_cfg or {}
    calibrator = StepwiseConformalCalibrator(
        delta=risk_delta,
        num_steps=num_steps,
        initial_quantile=float(calibrator_params.get("initial_quantile", 0.60)),
        ema_alpha=calibrator_params.get("ema_alpha"),
        clip_min=calibrator_params.get("clip_min"),
        clip_max=calibrator_params.get("clip_max"),
        low_support=int(calibrator_params.get("low_support", 50)),
        window_size=calibrator_params.get("window_size"),
        init_thresholds=calibrator_params.get("init_thresholds"),
    )
    budget_controller = BudgetController(budget_fraction=budget_fraction)
    rollback = RollbackBuffer()
    effective_budget_tokens: List[float] = []
    effective_budget_ops: List[float] = []

    watchdog_min_cos = cfg.watchdog_min_cos
    watchdog_max_kl = cfg.watchdog_max_kl

    per_seq_latency_ms: List[float] = []
    final_consistency: List[float] = []
    risk_threshold_history: List[float] = []
    delta_violation_history: List[float] = []
    applied_thresholds: Dict[int, Dict[str, float]] = {}
    total_start = time.time()
    sampler = GPUUtilSampler(interval_sec=0.5)
    sampler.start()

    is_wikitext = task_name == "wikitext"
    is_lambada = task_name == "lambada"
    is_gsm8k = task_name == "gsm8k"
    ppl_stats: Dict[str, float] = {"nll_sum": 0.0, "token_count": 0.0}
    lambada_correct = 0
    lambada_total = len(labels) if is_lambada and labels is not None else 0
    gsm_correct = 0
    gsm_total = len(labels) if is_gsm8k and labels is not None else 0

    for prompt_idx, prompt in enumerate(prompts):
        dream_completion: Optional[str] = None
        if is_gsm8k:
            completion_text, augmented_prompt = _dream_completion_and_prompt(engine, prompt, gen_max_new)
            if completion_text:
                dream_completion = completion_text
                prompt = augmented_prompt
        state = engine.encode_prompt(prompt)
        prev_hidden: Optional[List[np.ndarray]] = None
        prev_logits: Optional[np.ndarray] = None
        prev_aux: Optional[Dict] = None

        total_tokens_computed = 0
        total_tokens_possible = 0
        # New: track token×layer×step ops
        num_layers_seen: Optional[int] = None
        baseline_ops = 0
        actual_ops = 0
        skip_stats_rows: List[str] = []

        seq_start = time.time()
        for t in range(num_steps):
            if prev_hidden is None or prev_logits is None:
                out = engine.step(state, t)
                prev_hidden = [h.copy() for h in out.hidden_by_layer]
                prev_logits = out.logits.copy()
                prev_aux = out.aux if isinstance(out.aux, dict) else None
                seq_len = out.logits.shape[0]
                if num_layers_seen is None:
                    num_layers_seen = len(out.hidden_by_layer)
                total_tokens_computed += seq_len
                total_tokens_possible += seq_len
                baseline_ops += (num_layers_seen * seq_len)
                actual_ops += (num_layers_seen * seq_len)
                skip_stats_rows.append(f"{t},{seq_len},0,0.0,{num_layers_seen},0")
                continue

            # Get compute mask based on current freeze cooldowns
            compute_mask = gate.get_compute_mask(prev_logits.shape[0])
            mask_to_use: Optional[np.ndarray]
            if compute_mask is None or np.all(compute_mask):
                mask_to_use = None
            else:
                mask_to_use = compute_mask.astype(bool)

            prev_hidden_snapshot = [h.copy() for h in prev_hidden]
            prev_logits_snapshot = prev_logits.copy()

            rollback.save(state)
            out = engine.step(state, t, compute_mask=mask_to_use)
            try:
                sampler.sample_once()
            except Exception:
                pass

            obs = compute_token_observables(
                out.hidden_by_layer,
                prev_hidden_snapshot,
                out.logits,
                prev_logits_snapshot,
                t,
                num_steps,
                aux=out.aux if isinstance(out.aux, dict) else None,
            )
            risk_scores = compute_risk_score(obs.cosine_norm, obs.delta_l2_norm, obs.kl)

            watchdog_mask = (obs.cosine < watchdog_min_cos) | (obs.kl > watchdog_max_kl)
            watchdog_mask_snapshot = watchdog_mask.copy()
            if np.any(watchdog_mask):
                calibrator.register(step=t, scores=risk_scores, unsafe_mask=watchdog_mask)
                rollback.restore(state)
                mask_to_use = None
                out = engine.step(state, t, compute_mask=None)
                try:
                    sampler.sample_once()
                except Exception:
                    pass
                obs = compute_token_observables(
                    out.hidden_by_layer,
                    prev_hidden_snapshot,
                    out.logits,
                    prev_logits_snapshot,
                    t,
                    num_steps,
                    aux=out.aux if isinstance(out.aux, dict) else None,
                )
                risk_scores = compute_risk_score(obs.cosine_norm, obs.delta_l2_norm, obs.kl)
                watchdog_mask = np.zeros_like(risk_scores, dtype=bool)

            seq_len = out.logits.shape[0]
            frozen_tokens = 0 if mask_to_use is None else (seq_len - int(np.sum(mask_to_use)))
            frozen_ratio = float(frozen_tokens) / float(seq_len)
            total_tokens_computed += (seq_len - frozen_tokens)
            total_tokens_possible += seq_len
            if num_layers_seen is None:
                num_layers_seen = len(out.hidden_by_layer)
            baseline_ops += (num_layers_seen * seq_len)
            actual_ops += (num_layers_seen * (seq_len - frozen_tokens))
            skip_stats_rows.append(f"{t},{seq_len},{frozen_tokens},{frozen_ratio:.6f},{num_layers_seen},0")
            freeze_ratios_by_step[t].append(frozen_ratio)

            risk_threshold = calibrator.effective_threshold(step=t, candidate_scores=risk_scores)
            risk_threshold_history.append(risk_threshold)
            if risk_scores.size > 0:
                delta_violation_history.append(float(np.mean(risk_scores > risk_threshold)))
            cosine_override = stepwise_thresholds.cosine_for_step(t) if stepwise_thresholds else None
            delta_override = stepwise_thresholds.delta_for_step(t) if stepwise_thresholds else None
            margin_override = stepwise_thresholds.margin_for_step(t) if stepwise_thresholds else None
            if stepwise_thresholds is not None:
                applied_thresholds[t] = {
                    "cosine_tau": float(cosine_override if cosine_override is not None else cfg.cosine_tau),
                    "delta_l2_rho": float(delta_override if delta_override is not None else cfg.delta_l2_rho),
                    "gamma_margin": float(margin_override if margin_override is not None else cfg.gamma_margin),
                }

            result = gate.update(
                feats_cos=obs.cosine,
                feats_dl2=obs.delta_l2,
                feats_kl=obs.kl,
                entropy=obs.entropy,
                margin=obs.margin,
                risk_scores=risk_scores,
                risk_threshold=risk_threshold,
                num_tokens=out.logits.shape[0],
                compute_mask=mask_to_use,
                budget_controller=budget_controller,
                cosine_tau_override=cosine_override,
                delta_l2_override=delta_override,
                gamma_margin_override=margin_override,
            )

            if decision_logger is not None:
                freeze_mask = result.freeze_mask
                if freeze_mask is None:
                    freeze_mask = np.zeros(out.logits.shape[0], dtype=bool)
                counters_snapshot = result.counters
                for token_idx in range(out.logits.shape[0]):
                    decision_label = "freeze" if bool(freeze_mask[token_idx]) else "keep"
                    k_applied = cfg.freeze_K if decision_label == "freeze" else 0
                    decision_logger.log_decision(
                        prompt_id=prompt_idx,
                        step=t,
                        token_idx=token_idx,
                        decision=decision_label,
                        k_applied=k_applied,
                        risk=float(risk_scores[token_idx]),
                        risk_threshold=float(risk_threshold),
                        margin=float(obs.margin[token_idx]),
                        kl=float(obs.kl[token_idx]),
                        one_minus_cos=float(1.0 - obs.cosine[token_idx]),
                        delta_l2=float(obs.delta_l2[token_idx]),
                        delta_l2_norm=float(obs.delta_l2_norm[token_idx]),
                        sigma=float(obs.sigma[token_idx]),
                        layer_band="global",
                        profile=profile,
                        extras={
                            "watchdog_triggered": bool(watchdog_mask_snapshot[token_idx]),
                            "counter": int(counters_snapshot[token_idx]),
                        },
                    )

            prev_hidden = [h.copy() for h in out.hidden_by_layer]
            prev_logits = out.logits.copy()
            prev_aux = out.aux if isinstance(out.aux, dict) else None

        per_seq_latency_ms.append((time.time() - seq_start) * 1000.0)

        final_tokens: Optional[List[int]] = None
        final_margins: Optional[List[float]] = None
        decoded_text: Optional[str] = None
        if prev_logits is not None:
            if is_wikitext and prev_aux is not None:
                input_ids = prev_aux.get("input_ids") if isinstance(prev_aux, dict) else None
                special_mask = prev_aux.get("special_tokens_mask") if isinstance(prev_aux, dict) else None
                pad_token_id = int(prev_aux.get("pad_token_id", -1)) if isinstance(prev_aux, dict) else -1
                if input_ids is not None:
                    _accumulate_ppl(ppl_stats, prev_logits, input_ids, pad_token_id, special_mask)
            final_tokens, final_margins = logits_argmax_and_margin(prev_logits)
            decoded_text = _safe_decode(engine, final_tokens)
            diffusion_text = decoded_text
            eval_text_val: Optional[str] = None
            eval_text_source: Optional[str] = None
            if dream_completion:
                eval_text_val = dream_completion
                eval_text_source = "dream_completion"
            elif (is_lambada or is_gsm8k) and labels is not None and prompt_idx < len(labels):
                eval_text_val, eval_text_source = _evaluation_text_from_diffusion(
                    diffusion_text, prompt, engine, gen_max_new
                )
            if is_lambada and labels is not None and prompt_idx < len(labels):
                gold_word = labels[prompt_idx]
                gen_text = eval_text_val or ""
                pred_word = _extract_last_word(gen_text.split()[0] if gen_text else "")
                if _normalize_word(pred_word) == _normalize_word(str(gold_word)):
                    lambada_correct += 1
            if is_gsm8k and labels is not None and prompt_idx < len(labels):
                gold_ans = labels[prompt_idx]
                gen_text = eval_text_val or ""
                pred_ans = _normalize_gsm_answer(_extract_gsm_answer(gen_text))
                if pred_ans == _normalize_gsm_answer(str(gold_ans)):
                    gsm_correct += 1

        # Token savings (old): per-step token compute ratio
        token_savings = 1.0 - (total_tokens_computed / max(1, total_tokens_possible))
        # FLOPs-proxy savings (new): token×layer×step
        flops_savings = 1.0 - (actual_ops / max(1, baseline_ops))
        savings_ratios.append(float(flops_savings))
        effective_budget_tokens.append(1.0 - token_savings)
        effective_budget_ops.append(1.0 - flops_savings)
        aggregates.append({
            "mode": "rule_gate",
            "prompt_len": len(prompt),
            "token_savings": float(token_savings),
            "flops_savings": float(flops_savings),
        })

        # Write skip_stats.csv for this prompt
        stats_csv = os.path.join(out_dirs["runs"], f"skip_stats_{int(time.time())}.csv")
        with open(stats_csv, "w", encoding="utf-8") as f:
            f.write("step,seq_len,frozen_tokens,frozen_ratio,num_layers,skipped_layers\n")
            for row in skip_stats_rows:
                f.write(row + "\n")

        if decision_logger is not None and prev_logits is not None and final_tokens is not None and final_margins is not None:
            checksum = float(np.sum(prev_logits)) if prev_logits.size else 0.0
            decision_logger.log_final_output(
                prompt_id=prompt_idx,
                tokens=final_tokens,
                margins=final_margins,
                logits_checksum=checksum,
                extras={
                    "prompt_length": len(prompt),
                    "prompt": prompt,
                    "decoded_text": decoded_text,
                    "evaluation_text_source": eval_text_source,
                },
            )

        if consistency_check:
            # Run full-compute baseline for the same prompt and compare final token argmax
            baseline_state = engine.encode_prompt(prompt)
            baseline_prev_logits: Optional[np.ndarray] = None
            for t in range(num_steps):
                baseline_out = engine.step(baseline_state, t, compute_mask=None)
                baseline_prev_logits = baseline_out.logits
            if baseline_prev_logits is not None and prev_logits is not None:
                pred_a = np.argmax(prev_logits, axis=-1)
                pred_b = np.argmax(baseline_prev_logits, axis=-1)
                consistency = float(np.mean((pred_a == pred_b).astype(np.float32)))
                final_consistency.append(consistency)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_jsonl(os.path.join(out_dirs["runs"], f"rule_gate_{timestamp}.jsonl"), aggregates)
    if len(savings_ratios) > 0:
        save_hist(np.array(savings_ratios), os.path.join(out_dirs["figures"], f"rule_gate_savings_hist_{timestamp}.png"), title="Token compute savings ratio")

    try:
        from src.viz.rich import render_rule_gate_suite

        render_rule_gate_suite(
            out_dirs["figures"],
            timestamp,
            savings_ratios,
            freeze_ratios_by_step,
            risk_threshold_history,
            delta_violation_history,
            per_seq_latency_ms,
        )
    except Exception:
        pass

    thresholds_manifest_path: Optional[str] = None
    if stepwise_thresholds is not None:
        thresholds_manifest_path = os.path.join(
            out_dirs["runs"], f"rule_gate_stepwise_thresholds_{timestamp}.json"
        )
        with open(thresholds_manifest_path, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "source": stepwise_thresholds.source_path,
                    "applied": {str(k): v for k, v in sorted(applied_thresholds.items())},
                    "cosine_tau": [float(x) for x in stepwise_thresholds.cosine_tau],
                    "delta_l2_rho": [float(x) for x in stepwise_thresholds.delta_l2_rho],
                    "gamma_margin": None if stepwise_thresholds.gamma_margin is None else [float(x) for x in stepwise_thresholds.gamma_margin],
                },
                fp,
                indent=2,
            )

    total_time_s = max(1e-6, time.time() - total_start)
    sampler.stop()
    gpu = sampler.metrics or (query_gpu_utilization() or {})

    quality_metrics: Dict[str, Dict[str, float]] = {}
    if is_wikitext and ppl_stats["token_count"] > 0:
        ppl_value = float(np.exp(ppl_stats["nll_sum"] / ppl_stats["token_count"]))
        quality_metrics["perplexity"] = {
            "value": ppl_value,
            "token_count": int(ppl_stats["token_count"]),
            "nll_sum": float(ppl_stats["nll_sum"]),
        }
    if is_lambada and lambada_total:
        quality_metrics["accuracy"] = {
            "value": float(lambada_correct / lambada_total if lambada_total else 0.0),
            "correct": int(lambada_correct),
            "total": int(lambada_total),
        }
    if is_gsm8k and gsm_total:
        quality_metrics["exact_match"] = {
            "value": float(gsm_correct / gsm_total if gsm_total else 0.0),
            "correct": int(gsm_correct),
            "total": int(gsm_total),
        }

    summary = {
        "mode": "rule_gate",
        "num_prompts": len(prompts),
        "num_steps": num_steps,
        "latency_ms_mean": float(np.mean(per_seq_latency_ms) if per_seq_latency_ms else 0.0),
        "latency_ms_p50": float(np.percentile(per_seq_latency_ms, 50) if per_seq_latency_ms else 0.0),
        "latency_ms_p90": float(np.percentile(per_seq_latency_ms, 90) if per_seq_latency_ms else 0.0),
        "throughput_seq_per_s": float(len(prompts) / total_time_s),
        "skip_ratio_mean": float(np.mean(savings_ratios) if savings_ratios else 0.0),
        "final_token_consistency_mean": float(np.mean(final_consistency)) if final_consistency else None,
        "risk_threshold_mean": float(np.mean(risk_threshold_history)) if risk_threshold_history else None,
        "delta_violation_mean": float(np.mean(delta_violation_history)) if delta_violation_history else None,
        "budget_fraction": float(np.mean(effective_budget_ops)) if effective_budget_ops else None,
        "budget_fraction_target": float(budget_fraction),
        "budget_fraction_tokens": float(np.mean(effective_budget_tokens)) if effective_budget_tokens else None,
        "risk_delta": float(risk_delta),
        "stepwise_thresholds": thresholds_manifest_path,
        "stepwise_thresholds_source": stepwise_thresholds.source_path if stepwise_thresholds is not None else None,
        "decision_log": decision_logger.decision_path if decision_logger is not None else None,
        "final_outputs_path": decision_logger.final_outputs_path if decision_logger is not None else None,
        **{f"gpu_{k}": v for k, v in gpu.items()},
    }
    if quality_metrics:
        summary["quality"] = quality_metrics
    with open(os.path.join(out_dirs["runs"], f"rule_gate_summary_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def run_learned_gate(
    engine: BaseEngine,
    prompts: List[str],
    out_dirs: Dict[str, str],
    num_steps: int,
    gate: LearnedGate,
    watchdog_min_cos: float,
    watchdog_max_kl: float,
    risk_delta: float,
    budget_fraction: float,
    consistency_check: bool = False,
    calibrator_cfg: Optional[Dict[str, object]] = None,
    decision_logger: Optional[DecisionLogger] = None,
    profile: Optional[str] = None,
    labels: Optional[List[str]] = None,
    task_name: Optional[str] = None,
    gen_max_new: int = 128,
) -> None:
    from src.viz.plots import save_hist

    aggregates: List[Dict] = []
    savings_ratios: List[float] = []
    effective_budget_tokens: List[float] = []
    prob_samples: List[float] = []
    freeze_ratios_by_step: Dict[int, List[float]] = defaultdict(list)

    per_seq_latency_ms: List[float] = []
    final_consistency: List[float] = []
    total_start = time.time()
    sampler = GPUUtilSampler(interval_sec=0.5)
    sampler.start()
    calibrator_params = calibrator_cfg or {}
    calibrator = StepwiseConformalCalibrator(
        delta=risk_delta,
        num_steps=num_steps,
        initial_quantile=float(calibrator_params.get("initial_quantile", 0.60)),
        ema_alpha=calibrator_params.get("ema_alpha"),
        clip_min=calibrator_params.get("clip_min"),
        clip_max=calibrator_params.get("clip_max"),
        low_support=int(calibrator_params.get("low_support", 50)),
        window_size=calibrator_params.get("window_size"),
        init_thresholds=calibrator_params.get("init_thresholds"),
    )
    budget_controller = BudgetController(budget_fraction=budget_fraction)
    risk_threshold_history: List[float] = []
    delta_violation_history: List[float] = []

    is_wikitext = task_name == "wikitext"
    is_lambada = task_name == "lambada"
    is_gsm8k = task_name == "gsm8k"
    ppl_stats: Dict[str, float] = {"nll_sum": 0.0, "token_count": 0.0}
    lambada_correct = 0
    lambada_total = len(labels) if is_lambada and labels is not None else 0
    gsm_correct = 0
    gsm_total = len(labels) if is_gsm8k and labels is not None else 0

    for prompt_idx, prompt in enumerate(prompts):
        dream_completion: Optional[str] = None
        if is_gsm8k:
            completion_text, augmented_prompt = _dream_completion_and_prompt(engine, prompt, gen_max_new)
            if completion_text:
                dream_completion = completion_text
                prompt = augmented_prompt
        state = engine.encode_prompt(prompt)
        prev_hidden: Optional[List[np.ndarray]] = None
        prev_logits: Optional[np.ndarray] = None
        prev_aux: Optional[Dict] = None

        total_tokens_computed = 0
        total_tokens_possible = 0

        seq_start = time.time()
        for t in range(num_steps):
            if prev_hidden is None or prev_logits is None:
                out = engine.step(state, t)
                prev_hidden = [h.copy() for h in out.hidden_by_layer]
                prev_logits = out.logits.copy()
                prev_aux = out.aux if isinstance(out.aux, dict) else None
                total_tokens_computed += out.logits.shape[0]
                total_tokens_possible += out.logits.shape[0]
                continue

            compute_mask = gate.get_compute_mask(prev_logits.shape[0])
            mask_to_use: Optional[np.ndarray]
            if compute_mask is None or np.all(compute_mask):
                mask_to_use = None
            else:
                mask_to_use = compute_mask.astype(bool)

            prev_hidden_snapshot = [h.copy() for h in prev_hidden]
            prev_logits_snapshot = prev_logits.copy()

            out = engine.step(state, t, compute_mask=mask_to_use)

            seq_len = out.logits.shape[0]
            computed_tokens = seq_len if mask_to_use is None else int(np.sum(mask_to_use))
            total_tokens_computed += computed_tokens
            total_tokens_possible += seq_len

            obs = compute_token_observables(
                out.hidden_by_layer,
                prev_hidden_snapshot,
                out.logits,
                prev_logits_snapshot,
                t,
                num_steps,
                aux=out.aux if isinstance(out.aux, dict) else None,
            )
            risk_scores = compute_risk_score(obs.cosine_norm, obs.delta_l2_norm, obs.kl)

            watchdog_mask = (obs.cosine < watchdog_min_cos) | (obs.kl > watchdog_max_kl)
            # Snapshot before any potential reset so decision logs reflect the initial watchdog state
            watchdog_mask_snapshot = watchdog_mask.copy()
            if np.any(watchdog_mask):
                calibrator.register(step=t, scores=risk_scores, unsafe_mask=watchdog_mask)
                mask_to_use = None
                out = engine.step(state, t, compute_mask=None)
                obs = compute_token_observables(
                    out.hidden_by_layer,
                    prev_hidden_snapshot,
                    out.logits,
                    prev_logits_snapshot,
                    t,
                    num_steps,
                    aux=out.aux if isinstance(out.aux, dict) else None,
                )
                risk_scores = compute_risk_score(obs.cosine_norm, obs.delta_l2_norm, obs.kl)

            computed_tokens = out.logits.shape[0] if mask_to_use is None else int(np.sum(mask_to_use))
            total_tokens_computed += computed_tokens
            total_tokens_possible += out.logits.shape[0]
            # Track freeze ratio per step after compute mask is finalized
            seq_len = out.logits.shape[0]
            if seq_len > 0:
                freeze_ratios_by_step[t].append(1.0 - (computed_tokens / float(seq_len)))

            # Use raw features to match training FEATURE_NAMES[:7]
            features = np.stack(
                [
                    obs.cosine,
                    obs.delta_l2,
                    obs.kl,
                    obs.entropy,
                    obs.margin,
                    obs.step_frac,
                    obs.sigma,
                ],
                axis=-1,
            ).astype(np.float32)
            risk_threshold = calibrator.effective_threshold(step=t, candidate_scores=risk_scores)
            risk_threshold_history.append(risk_threshold)
            if risk_scores.size > 0:
                delta_violation_history.append(float(np.mean(risk_scores > risk_threshold)))
            gate_result = gate.update(
                features=features,
                num_tokens=out.logits.shape[0],
                risk_threshold=risk_threshold,
                feature_margin=obs.margin,
                compute_mask=mask_to_use,
                budget_controller=budget_controller,
            )

            if gate_result.probabilities is not None:
                prob_samples.extend(gate_result.probabilities.tolist())

            if decision_logger is not None:
                freeze_mask = gate_result.freeze_mask
                if freeze_mask is None:
                    freeze_mask = np.zeros(out.logits.shape[0], dtype=bool)
                probabilities = gate_result.probabilities if gate_result.probabilities is not None else np.zeros(out.logits.shape[0], dtype=np.float32)
                counters_snapshot = gate_result.counters
                for token_idx in range(out.logits.shape[0]):
                    decision_label = "freeze" if bool(freeze_mask[token_idx]) else "keep"
                    k_applied = gate.cfg.freeze_K if decision_label == "freeze" else 0
                    decision_logger.log_decision(
                        prompt_id=prompt_idx,
                        step=t,
                        token_idx=token_idx,
                        decision=decision_label,
                        k_applied=k_applied,
                        risk=float(risk_scores[token_idx]),
                        risk_threshold=float(risk_threshold),
                        margin=float(obs.margin[token_idx]),
                        kl=float(obs.kl[token_idx]),
                        one_minus_cos=float(1.0 - obs.cosine[token_idx]),
                        delta_l2=float(obs.delta_l2[token_idx]),
                        delta_l2_norm=float(obs.delta_l2_norm[token_idx]),
                        sigma=float(obs.sigma[token_idx]),
                        layer_band="global",
                        profile=profile,
                        extras={
                            "watchdog_triggered": bool(watchdog_mask_snapshot[token_idx]),
                            "probability": float(probabilities[token_idx]),
                            "counter": int(counters_snapshot[token_idx]),
                        },
                    )

            prev_hidden = [h.copy() for h in out.hidden_by_layer]
            prev_logits = out.logits.copy()
            prev_aux = out.aux if isinstance(out.aux, dict) else None

        per_seq_latency_ms.append((time.time() - seq_start) * 1000.0)

        savings = 1.0 - (total_tokens_computed / max(1, total_tokens_possible))
        savings_ratios.append(float(savings))
        effective_budget_tokens.append(1.0 - savings)
        aggregates.append({"mode": "learned_gate", "prompt_len": len(prompt), "savings": float(savings)})

        final_tokens: Optional[List[int]] = None
        final_margins: Optional[List[float]] = None
        decoded_text: Optional[str] = None
        eval_text_source: Optional[str] = None
        if prev_logits is not None:
            if is_wikitext and prev_aux is not None:
                input_ids = prev_aux.get("input_ids") if isinstance(prev_aux, dict) else None
                special_mask = prev_aux.get("special_tokens_mask") if isinstance(prev_aux, dict) else None
                pad_token_id = int(prev_aux.get("pad_token_id", -1)) if isinstance(prev_aux, dict) else -1
                if input_ids is not None:
                    _accumulate_ppl(ppl_stats, prev_logits, input_ids, pad_token_id, special_mask)
            final_tokens, final_margins = logits_argmax_and_margin(prev_logits)
            decoded_text = _safe_decode(engine, final_tokens)
            diffusion_text = decoded_text
            eval_text_val: Optional[str] = None
            if dream_completion:
                eval_text_val = dream_completion
                eval_text_source = "dream_completion"
            elif (is_lambada or is_gsm8k) and labels is not None and prompt_idx < len(labels):
                eval_text_val, eval_text_source = _evaluation_text_from_diffusion(
                    diffusion_text, prompt, engine, gen_max_new
                )
            if is_lambada and labels is not None and prompt_idx < len(labels):
                gold_word = labels[prompt_idx]
                gen_text = eval_text_val or ""
                pred_word = _extract_last_word(gen_text.split()[0] if gen_text else "")
                if _normalize_word(pred_word) == _normalize_word(str(gold_word)):
                    lambada_correct += 1
            if is_gsm8k and labels is not None and prompt_idx < len(labels):
                gold_ans = labels[prompt_idx]
                gen_text = eval_text_val or ""
                pred_ans = _normalize_gsm_answer(_extract_gsm_answer(gen_text))
                if pred_ans == _normalize_gsm_answer(str(gold_ans)):
                    gsm_correct += 1

        if decision_logger is not None and prev_logits is not None and final_tokens is not None and final_margins is not None:
            checksum = float(np.sum(prev_logits)) if prev_logits.size else 0.0
            decision_logger.log_final_output(
                prompt_id=prompt_idx,
                tokens=final_tokens,
                margins=final_margins,
                logits_checksum=checksum,
                extras={
                    "prompt_length": len(prompt),
                    "prompt": prompt,
                    "decoded_text": decoded_text,
                    "evaluation_text_source": eval_text_source,
                },
            )

        if consistency_check:
            baseline_state = engine.encode_prompt(prompt)
            baseline_prev_logits: Optional[np.ndarray] = None
            for t in range(num_steps):
                baseline_out = engine.step(baseline_state, t, compute_mask=None)
                baseline_prev_logits = baseline_out.logits
            if baseline_prev_logits is not None and prev_logits is not None:
                pred_a = np.argmax(prev_logits, axis=-1)
                pred_b = np.argmax(baseline_prev_logits, axis=-1)
                consistency = float(np.mean((pred_a == pred_b).astype(np.float32)))
                final_consistency.append(consistency)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_jsonl(os.path.join(out_dirs["runs"], f"learned_gate_{timestamp}.jsonl"), aggregates)
    if savings_ratios:
        save_hist(np.array(savings_ratios), os.path.join(out_dirs["figures"], f"learned_gate_savings_hist_{timestamp}.png"), title="Token compute savings ratio (learned gate)")
    if prob_samples:
        save_hist(np.array(prob_samples), os.path.join(out_dirs["figures"], f"learned_gate_prob_hist_{timestamp}.png"), title="Learned gate freeze probability")

    try:
        from src.viz.rich import render_learned_gate_suite

        render_learned_gate_suite(
            out_dirs["figures"],
            timestamp,
            savings_ratios,
            freeze_ratios_by_step,
            risk_threshold_history,
            delta_violation_history,
            per_seq_latency_ms,
            prob_samples,
        )
    except Exception:
        pass

    total_time_s = max(1e-6, time.time() - total_start)
    sampler.stop()
    gpu = sampler.metrics or (query_gpu_utilization() or {})

    quality_metrics: Dict[str, Dict[str, float]] = {}
    if is_wikitext and ppl_stats["token_count"] > 0:
        ppl_value = float(np.exp(ppl_stats["nll_sum"] / ppl_stats["token_count"]))
        quality_metrics["perplexity"] = {
            "value": ppl_value,
            "token_count": int(ppl_stats["token_count"]),
            "nll_sum": float(ppl_stats["nll_sum"]),
        }
    if is_lambada and lambada_total:
        quality_metrics["accuracy"] = {
            "value": float(lambada_correct / lambada_total if lambada_total else 0.0),
            "correct": int(lambada_correct),
            "total": int(lambada_total),
        }
    if is_gsm8k and gsm_total:
        quality_metrics["exact_match"] = {
            "value": float(gsm_correct / gsm_total if gsm_total else 0.0),
            "correct": int(gsm_correct),
            "total": int(gsm_total),
        }

    summary = {
        "mode": "learned_gate",
        "num_prompts": len(prompts),
        "num_steps": num_steps,
        "latency_ms_mean": float(np.mean(per_seq_latency_ms) if per_seq_latency_ms else 0.0),
        "latency_ms_p50": float(np.percentile(per_seq_latency_ms, 50) if per_seq_latency_ms else 0.0),
        "latency_ms_p90": float(np.percentile(per_seq_latency_ms, 90) if per_seq_latency_ms else 0.0),
        "throughput_seq_per_s": float(len(prompts) / total_time_s),
        "skip_ratio_mean": float(np.mean(savings_ratios) if savings_ratios else 0.0),
        "final_token_consistency_mean": float(np.mean(final_consistency)) if final_consistency else None,
        "risk_threshold_mean": float(np.mean(risk_threshold_history)) if risk_threshold_history else None,
        "delta_violation_mean": float(np.mean(delta_violation_history)) if delta_violation_history else None,
        "budget_fraction": float(np.mean(effective_budget_tokens)) if effective_budget_tokens else None,
        "budget_fraction_target": float(budget_fraction),
        "risk_delta": float(risk_delta),
        "decision_log": decision_logger.decision_path if decision_logger is not None else None,
        "final_outputs_path": decision_logger.final_outputs_path if decision_logger is not None else None,
        **{f"gpu_{k}": v for k, v in gpu.items()},
    }
    if quality_metrics:
        summary["quality"] = quality_metrics
    with open(os.path.join(out_dirs["runs"], f"learned_gate_summary_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def run_adaptive(
    engine: BaseEngine,
    prompts: List[str],
    out_dirs: Dict[str, str],
    num_steps: int,
    scheduler_cfg: AdaptiveSchedulerConfig,
    risk_delta: float,
    skip_budget: float,
    hard_budget: bool = False,
    labels: Optional[List[str]] = None,
    task_name: Optional[str] = None,
    gen_max_new: int = 128,
    decision_logger: Optional[DecisionLogger] = None,
) -> None:
    scheduler = AdaptiveScheduler(scheduler_cfg)
    calibrator = ConformalRiskCalibrator(delta=risk_delta)
    per_seq_latency_ms: List[float] = []
    skip_estimates: List[float] = []
    lte_traces: List[List[float]] = []
    risk_traces: List[List[float]] = []
    stride_traces: List[List[int]] = []
    lte_threshold_traces: List[List[float]] = []
    total_start = time.time()
    sampler = GPUUtilSampler(interval_sec=0.5)
    sampler.start()

    is_wikitext = task_name == "wikitext"
    is_lambada = task_name == "lambada"
    is_gsm8k = task_name == "gsm8k"
    ppl_stats: Dict[str, float] = {"nll_sum": 0.0, "token_count": 0.0}
    lambada_correct = 0
    lambada_total = len(labels) if is_lambada and labels is not None else 0
    gsm_correct = 0
    gsm_total = len(labels) if is_gsm8k and labels is not None else 0

    for prompt_idx, prompt in enumerate(prompts):
        dream_completion: Optional[str] = None
        if is_gsm8k:
            completion_text, augmented_prompt = _dream_completion_and_prompt(engine, prompt, gen_max_new)
            if completion_text:
                dream_completion = completion_text
                prompt = augmented_prompt
        state = engine.encode_prompt(prompt)
        prev_hidden: Optional[List[np.ndarray]] = None
        prev_logits: Optional[np.ndarray] = None
        prev_aux: Optional[Dict] = None
        final_tokens: Optional[List[int]] = None
        final_margins: Optional[List[float]] = None
        decoded_text: Optional[str] = None
        seq_start = time.time()
        effective_steps = 0.0
        prompt_lte: List[float] = []
        prompt_risk: List[float] = []
        prompt_stride: List[int] = []
        prompt_lte_thresholds: List[float] = []
        lte_history: List[float] = []

        for t in range(num_steps):
            out = engine.step(state, t, compute_mask=None)
            if prev_hidden is None or prev_logits is None:
                prev_hidden = [h.copy() for h in out.hidden_by_layer]
                prev_logits = out.logits.copy()
                prev_aux = out.aux if isinstance(out.aux, dict) else None
                effective_steps += 1.0
                prompt_stride.append(scheduler.stride)
                continue

            obs = compute_token_observables(
                out.hidden_by_layer,
                prev_hidden,
                out.logits,
                prev_logits,
                t,
                num_steps,
                aux=out.aux if isinstance(out.aux, dict) else None,
            )
            risk_scores = compute_risk_score(obs.cosine_norm, obs.delta_l2_norm, obs.kl)
            calibrator.register(risk_scores, unsafe_mask=None)
            risk_threshold = calibrator.effective_threshold(risk_scores)
            lte_raw = float(np.mean(np.abs(obs.delta_l2)))
            lte_norm = float(np.mean(np.abs(obs.delta_l2_norm)))
            lte_value = lte_norm if scheduler_cfg.use_normalized_lte else lte_raw
            risk_metric = float(np.quantile(risk_scores.astype(np.float64), 0.90, method="higher")) if risk_scores.size > 0 else 0.0
            lte_history.append(lte_value)
            if scheduler_cfg.use_normalized_lte or scheduler_cfg.lte_eps > 1.0:
                lte_threshold = float(scheduler_cfg.lte_eps)
            else:
                quantile = float(np.clip(scheduler_cfg.lte_eps, 0.0, 1.0))
                lte_threshold = float(np.quantile(np.asarray(lte_history, dtype=np.float32), quantile))
            scheduler.observe(lte_value, lte_threshold, risk_metric, risk_threshold)
            prompt_lte.append(lte_value)
            prompt_risk.append(risk_metric)
            prompt_stride.append(scheduler.stride)
            prompt_lte_thresholds.append(lte_threshold)
            stride = scheduler.stride
            # Optional hard cap on cumulative skip ratio per sequence
            if hard_budget and (t + 1) > 0:
                eff_next = effective_steps + (1.0 / max(1, stride))
                skip_so_far = 1.0 - (eff_next / float(t + 1))
                if skip_so_far > skip_budget:
                    stride = 1
                    # reflect the correction in recorded stride and internal state
                    prompt_stride[-1] = 1
                    try:
                        scheduler._current_stride = 1  # best-effort sync
                    except Exception:
                        pass
            effective_steps += 1.0 / max(1, stride)

            prev_hidden = [h.copy() for h in out.hidden_by_layer]
            prev_logits = out.logits.copy()
            prev_aux = out.aux if isinstance(out.aux, dict) else None

        per_seq_latency_ms.append((time.time() - seq_start) * 1000.0)
        if prev_logits is not None:
            if is_wikitext and prev_aux is not None:
                input_ids = prev_aux.get("input_ids") if isinstance(prev_aux, dict) else None
                special_mask = prev_aux.get("special_tokens_mask") if isinstance(prev_aux, dict) else None
                pad_token_id = int(prev_aux.get("pad_token_id", -1)) if isinstance(prev_aux, dict) else -1
                if input_ids is not None:
                    _accumulate_ppl(ppl_stats, prev_logits, input_ids, pad_token_id, special_mask)
            tokens, margins = logits_argmax_and_margin(prev_logits)
            decoded_text = _safe_decode(engine, tokens)
            diffusion_text = decoded_text
            final_tokens = tokens
            final_margins = margins
            eval_text_source: Optional[str] = None
            eval_text_val: Optional[str] = None
            if dream_completion:
                eval_text_val = dream_completion
                eval_text_source = "dream_completion"
            elif (is_lambada or is_gsm8k) and labels is not None and prompt_idx < len(labels):
                eval_text_val, eval_text_source = _evaluation_text_from_diffusion(
                    diffusion_text, prompt, engine, gen_max_new
                )
            if is_lambada and labels is not None and prompt_idx < len(labels):
                gold_word = labels[prompt_idx]
                gen_text = eval_text_val or ""
                pred_word = _extract_last_word(gen_text.split()[0] if gen_text else "")
                if _normalize_word(pred_word) == _normalize_word(str(gold_word)):
                    lambada_correct += 1
            if is_gsm8k and labels is not None and prompt_idx < len(labels):
                gold_ans = labels[prompt_idx]
                gen_text = eval_text_val or ""
                pred_ans = _normalize_gsm_answer(_extract_gsm_answer(gen_text))
                if pred_ans == _normalize_gsm_answer(str(gold_ans)):
                    gsm_correct += 1
        nominal_steps = float(num_steps)
        skip_estimate = max(0.0, 1.0 - (effective_steps / nominal_steps))
        skip_estimates.append(skip_estimate)
        scheduler.reset()
        lte_traces.append(prompt_lte)
        risk_traces.append(prompt_risk)
        stride_traces.append(prompt_stride)
        lte_threshold_traces.append(prompt_lte_thresholds)
        if decision_logger is not None and prev_logits is not None and final_tokens is not None and final_margins is not None:
            checksum = float(np.sum(prev_logits)) if prev_logits.size else 0.0
            decision_logger.log_final_output(
                prompt_id=prompt_idx,
                tokens=final_tokens,
                margins=final_margins,
                logits_checksum=checksum,
                extras={
                    "prompt_length": len(prompt),
                    "prompt": prompt,
                    "decoded_text": decoded_text,
                    "stride_trace": prompt_stride,
                    "lte_trace": prompt_lte,
                    "risk_trace": prompt_risk,
                    "lte_threshold_trace": prompt_lte_thresholds,
                    "skip_estimate": skip_estimate,
                },
            )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Simple visuals: histogram of per-sequence estimated skip ratios
    try:
        from src.viz.plots import save_hist
        if skip_estimates:
            save_hist(np.array(skip_estimates), os.path.join(out_dirs["figures"], f"adaptive_skip_hist_{timestamp}.png"), title="Adaptive estimated skip ratio per sequence")
    except Exception:
        pass
    try:
        from src.viz.rich import render_adaptive_suite
        render_adaptive_suite(
            out_dirs["figures"],
            timestamp,
            skip_estimates,
            lte_traces,
            risk_traces,
            stride_traces,
            lte_threshold_traces,
            scheduler_cfg,
        )
    except Exception:
        pass
    total_time_s = max(1e-6, time.time() - total_start)
    sampler.stop()
    gpu = sampler.metrics or (query_gpu_utilization() or {})
    quality_metrics: Dict[str, Dict[str, float]] = {}
    if is_wikitext and ppl_stats["token_count"] > 0:
        ppl_value = float(np.exp(ppl_stats["nll_sum"] / ppl_stats["token_count"]))
        quality_metrics["perplexity"] = {
            "value": ppl_value,
            "token_count": int(ppl_stats["token_count"]),
            "nll_sum": float(ppl_stats["nll_sum"]),
        }
    if is_lambada and lambada_total:
        quality_metrics["accuracy"] = {
            "value": float(lambada_correct / lambada_total if lambada_total else 0.0),
            "correct": int(lambada_correct),
            "total": int(lambada_total),
        }
    if is_gsm8k and gsm_total:
        quality_metrics["exact_match"] = {
            "value": float(gsm_correct / gsm_total if gsm_total else 0.0),
            "correct": int(gsm_correct),
            "total": int(gsm_total),
        }
    summary = {
        "mode": "adaptive",
        "num_prompts": len(prompts),
        "num_steps": num_steps,
        "latency_ms_mean": float(np.mean(per_seq_latency_ms) if per_seq_latency_ms else 0.0),
        "latency_ms_p50": float(np.percentile(per_seq_latency_ms, 50) if per_seq_latency_ms else 0.0),
        "latency_ms_p90": float(np.percentile(per_seq_latency_ms, 90) if per_seq_latency_ms else 0.0),
        "throughput_seq_per_s": float(len(prompts) / total_time_s) if len(prompts) else 0.0,
        "skip_ratio_est_mean": float(np.mean(skip_estimates) if skip_estimates else 0.0),
        "skip_budget": float(skip_budget),
        "risk_delta": float(risk_delta),
        "lte_eps": float(scheduler_cfg.lte_eps),
        "max_stride": int(scheduler_cfg.max_stride),
        "decision_log": decision_logger.decision_path if decision_logger is not None else None,
        "final_outputs_path": decision_logger.final_outputs_path if decision_logger is not None else None,
        **{f"gpu_{k}": v for k, v in gpu.items()},
    }
    if quality_metrics:
        summary["quality"] = quality_metrics
    with open(os.path.join(out_dirs["runs"], f"adaptive_summary_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def run_baselines(
    out_dirs: Dict[str, str],
    num_steps: int,
    skip_ratios: Optional[List[float]] = None,
) -> None:
    baselines: List[Dict[str, float]] = []

    for t0 in (4, 6, 8):
        for stride in (2, 3):
            if t0 >= num_steps:
                skip = 0.0
            else:
                active_steps = max(0, num_steps - t0)
                skip = (active_steps / max(1, num_steps)) * (1.0 - (1.0 / float(stride)))
            baselines.append({
                "name": f"fixed_stride_t{t0}_s{stride}",
                "skip_ratio_est": skip,
            })

    # Index-only policy: stride increases with step index
    index_skip = 0.0
    for idx in range(1, num_steps + 1):
        stride = 1 + (idx // 4)
        index_skip += 1.0 - (1.0 / float(stride))
    baselines.append({"name": "index_only", "skip_ratio_est": index_skip / max(1, num_steps)})

    # Uniform layer throttling approximated as constant fraction savings
    for M in (2, 3, 4):
        skip = 1.0 - (1.0 / float(M))
        baselines.append({"name": f"layer_throttle_M{M}", "skip_ratio_est": skip})

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(out_dirs["runs"], f"baselines_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump({"num_steps": num_steps, "baselines": baselines}, f, indent=2)
    # Also write a quick histogram for visualization
    try:
        from src.viz.plots import save_hist
        save_hist(
            np.array([b["skip_ratio_est"] for b in baselines], dtype=np.float32),
            os.path.join(out_dirs["figures"], f"baselines_skip_hist_{timestamp}.png"),
            title="Baseline estimated skip ratios",
        )
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    default_paths = load_default_paths(coalesce(args.paths_cfg, cfg.get("paths_config")))

    repo_root = Path(__file__).resolve().parents[1]
    mode = args.mode

    phase = coalesce(args.phase, cfg.get("phase"), "P1")
    engine_name = coalesce(args.engine, cfg.get("engine"), "mock")
    outputs_root = coalesce(
        args.outputs_root,
        args.out_dir,
        cfg.get("outputs_root"),
        cfg.get("out_dir"),
        default_paths.get("outputs_root"),
        str((repo_root / "reports").resolve()),
    )
    data_root = coalesce(args.data_root, cfg.get("data_root"), default_paths.get("data_root"))
    models_root = coalesce(args.models_root, cfg.get("models_root"), default_paths.get("models_root"))
    exp_name = coalesce(args.exp_name, cfg.get("exp_name"))
    num_prompts = int(coalesce(args.num_prompts, cfg.get("num_prompts"), 8))
    max_new_tokens = int(coalesce(args.max_new_tokens, cfg.get("max_new_tokens"), 64))
    num_steps = int(coalesce(args.num_steps, cfg.get("num_steps"), 12))
    task = coalesce(args.task, cfg.get("task"), "synthetic")

    dump_features_raw = coalesce(args.dump_features_dir, cfg.get("dump_features_dir"))
    label_cfg = cfg.get("label_thresholds", {})
    label_thresholds = {
        "cosine_tau": float(coalesce(args.label_cos_tau, label_cfg.get("cosine_tau"), 0.97)),
        "delta_l2_rho": float(coalesce(args.label_dl2_rho, label_cfg.get("delta_l2_rho"), 0.05)),
        "kl_max": float(coalesce(args.label_kl_max, label_cfg.get("kl_max"), 0.01)),
    }

    tau = float(coalesce(args.tau, cfg.get("tau"), 0.95))
    rho = float(coalesce(args.rho, cfg.get("rho"), 0.10))
    consecutive_m = int(coalesce(args.m, cfg.get("m"), 2))
    freeze_K = int(coalesce(args.freeze_K, cfg.get("freeze_K"), 2))
    layer_recompute_M = int(coalesce(args.layer_recompute_M, cfg.get("layer_recompute_M"), 2))
    stepwise_cfg = cfg.get("stepwise_thresholds", {})
    gamma_margin = float(coalesce(args.gamma_margin, cfg.get("gamma_margin"), stepwise_cfg.get("margin_gamma"), 0.08))
    watchdog_cfg = cfg.get("watchdog", {})
    watchdog_min_cos = float(coalesce(args.watchdog_min_cos, watchdog_cfg.get("min_cos"), 0.90))
    watchdog_max_kl = float(coalesce(args.watchdog_max_kl, watchdog_cfg.get("max_kl"), 0.02))
    risk_delta = float(coalesce(args.risk_delta, cfg.get("risk_delta"), 0.01))
    budget_fraction = float(coalesce(args.budget_fraction, cfg.get("budget_fraction"), 1.0))
    quantiles_path_raw = coalesce(args.quantiles_path, cfg.get("quantiles_path"), stepwise_cfg.get("quantiles_path"))
    cos_quantile_level_raw = coalesce(args.one_minus_cos_quantile, cfg.get("one_minus_cos_quantile"), stepwise_cfg.get("one_minus_cos_q"))
    delta_quantile_level_raw = coalesce(args.delta_l2_quantile, cfg.get("delta_l2_quantile"), stepwise_cfg.get("dl2_norm_q"))
    kl_quantile_level_raw = coalesce(args.kl_quantile, cfg.get("kl_quantile"), stepwise_cfg.get("kl_q"))
    stepwise_ema_raw = coalesce(args.stepwise_ema, cfg.get("stepwise_ema"), stepwise_cfg.get("ema"))
    clip_min, clip_max = parse_clip_config(coalesce(args.stepwise_clip, cfg.get("stepwise_clip"), stepwise_cfg.get("clip_quantiles")))
    stepwise_ema = float(stepwise_ema_raw) if stepwise_ema_raw is not None else None
    cos_quantile_level = float(cos_quantile_level_raw) if cos_quantile_level_raw is not None else None
    delta_quantile_level = float(delta_quantile_level_raw) if delta_quantile_level_raw is not None else None
    kl_quantile_level = float(kl_quantile_level_raw) if kl_quantile_level_raw is not None else None
    quantiles_path = None
    if quantiles_path_raw:
        quantiles_path = quantiles_path_raw if os.path.isabs(quantiles_path_raw) else os.path.abspath(os.path.join(repo_root, quantiles_path_raw))
    risk_quantiles_path_raw = coalesce(args.risk_quantiles_path, cfg.get("risk_quantiles_path"), stepwise_cfg.get("risk_quantiles_path"))
    risk_initial_quantile_raw = coalesce(args.risk_initial_quantile, cfg.get("risk_initial_quantile"), stepwise_cfg.get("risk_initial_quantile"), 0.60)
    risk_initial_quantile = float(risk_initial_quantile_raw) if risk_initial_quantile_raw is not None else 0.60
    risk_low_support_raw = coalesce(args.risk_low_support, cfg.get("risk_low_support"), stepwise_cfg.get("risk_low_support"), 50)
    risk_low_support = int(risk_low_support_raw) if risk_low_support_raw is not None else 50
    risk_window_raw = coalesce(args.risk_window, cfg.get("risk_window"), stepwise_cfg.get("risk_window"))
    risk_window = int(risk_window_raw) if risk_window_raw else None
    risk_quantiles_path = None
    if risk_quantiles_path_raw:
        risk_quantiles_path = risk_quantiles_path_raw if os.path.isabs(risk_quantiles_path_raw) else os.path.abspath(os.path.join(repo_root, risk_quantiles_path_raw))
    risk_init_thresholds = None
    if risk_quantiles_path and os.path.exists(risk_quantiles_path):
        with open(risk_quantiles_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        steps_block = data.get("steps") if isinstance(data, dict) else None
        if isinstance(steps_block, dict):
            risk_init_thresholds = {int(step): float(val) for step, val in steps_block.items() if val is not None}
        else:
            risk_init_thresholds = {int(step): float(val) for step, val in data.items() if val is not None}

    calibrator_cfg = {
        "initial_quantile": risk_initial_quantile,
        "ema_alpha": stepwise_ema,
        "clip_min": clip_min,
        "clip_max": clip_max,
        "low_support": risk_low_support,
        "window_size": risk_window,
        "init_thresholds": risk_init_thresholds,
    }
    risk_manifest_cfg = dict(calibrator_cfg)
    risk_manifest_cfg["risk_quantiles_path"] = risk_quantiles_path

    outputs_root = str(outputs_root)
    run_dir = ensure_output_dirs(outputs_root, phase=phase, mode=mode, engine=engine_name, exp_name=exp_name)
    init_run_logging(run_dir, f"{mode}-{engine_name}")

    dump_features_dir = None
    if dump_features_raw:
        dump_features_dir = dump_features_raw if os.path.isabs(dump_features_raw) else os.path.join(run_dir, dump_features_raw)

    resolved_manifest = {
        "phase": phase,
        "mode": mode,
        "engine": engine_name,
        "outputs_root": outputs_root,
        "data_root": data_root,
        "models_root": models_root,
        "num_prompts": num_prompts,
        "max_new_tokens": max_new_tokens,
        "num_steps": num_steps,
        "task": task,
        "label_thresholds": label_thresholds,
        "rule_gate": {
            "tau": tau,
            "rho": rho,
            "m": consecutive_m,
            "freeze_K": freeze_K,
            "layer_recompute_M": layer_recompute_M,
            "gamma_margin": gamma_margin,
            "watchdog_min_cos": watchdog_min_cos,
            "watchdog_max_kl": watchdog_max_kl,
        },
        "stepwise_thresholds": {
            "quantiles_path": quantiles_path,
            "one_minus_cos_quantile": cos_quantile_level,
            "delta_l2_quantile": delta_quantile_level,
            "kl_quantile": kl_quantile_level,
            "ema": stepwise_ema,
            "clip_min": clip_min,
            "clip_max": clip_max,
        },
        "risk_calibrator": risk_manifest_cfg,
    }
    save_manifest(run_dir, {
        "args": vars(args),
        "config": cfg,
        "resolved": resolved_manifest,
    })

    wandb_run = None
    if wandb is not None:
        try:
            os.environ.setdefault("WANDB_MODE", "online")
            try:
                wandb.require("core")
            except Exception:
                pass
            wandb_run = wandb.init(project=os.environ.get("WANDB_PROJECT", "dcllm-delta"), name=os.path.basename(run_dir), dir=run_dir)
        except Exception as exc:  # pragma: no cover
            print(f"[wandb] init failed: {exc}; continuing without W&B logging.")
    else:
        print("[wandb] Package not installed; skipping W&B logging.")

    if engine_name == "mock":
        engine: BaseEngine = MockDiffusionEngine(
            vocab_size=1000,
            num_layers=4,
            hidden_size=64,
            sequence_length=max_new_tokens,
            num_steps=num_steps,
            rng_seed=42,
        )
    elif engine_name == "d2f":
        if D2FDreamEngine is None:
            raise ImportError("D2F engine requested but dependencies are missing. Install torch/transformers and retry.")
        default_model_path = str(Path(models_root) / "checkpoints" / "Dream-v0-Instruct-7B") if models_root else None
        default_lora_path = str(Path(models_root) / "lora" / "D2F_Dream_Base_7B_Lora") if models_root else None
        model_path = coalesce(args.model_path, cfg.get("model_path"), default_model_path)
        lora_path = coalesce(args.lora_path, cfg.get("lora_path"), default_lora_path)
        use_lora = bool(args.use_d2f_lora or cfg.get("use_d2f_lora", False))
        # Allow a longer prompt context to avoid truncating few-shot/chat templates
        prompt_max_len = 2048
        engine = D2FDreamEngine(
            model_name="d2f-small",
            device="cuda",
            model_path=model_path,
            lora_path=lora_path,
            use_lora=use_lora,
            num_steps=num_steps,
            max_seq_len=max_new_tokens,
            prompt_max_len=prompt_max_len,
        )
        # Pass layer reuse cadence into engine (optional)
        try:
            engine.layer_recompute_M = int(coalesce(args.layer_recompute_M, cfg.get("layer_recompute_M"), 0) or 0)
        except Exception:
            engine.layer_recompute_M = 0
    else:
        raise ValueError(f"Unsupported engine: {engine_name}")

    prompts, prompt_labels = load_prompts(num_prompts, task, data_root)
    out_dirs = {"figures": os.path.join(run_dir, "figures"), "runs": os.path.join(run_dir, "runs")}

    decision_loggers: List[DecisionLogger] = []
    decision_logger: Optional[DecisionLogger] = None
    if mode in {"rule_gate", "learned_gate", "adaptive"}:
        decision_logger = DecisionLogger(run_dir, mode, os.path.basename(run_dir))
        decision_loggers.append(decision_logger)

    teacher_artifacts: Optional[TeacherArtifacts] = None
    try:
        with Timer(f"run-{mode}"):
            if mode == "teacher":
                teacher_artifacts = run_teacher(
                    engine,
                    prompts,
                    out_dirs,
                    num_steps,
                    dump_features_dir=dump_features_dir,
                    label_thresholds=label_thresholds,
                    oracle_eval=bool(args.oracle_eval or cfg.get("oracle_eval", False)),
                    task_name=task,
                    exp_name=exp_name,
                    labels=prompt_labels,
                    gen_max_new=max_new_tokens,
                )
                if args.consistency_full_compute and engine_name == "d2f":
                    # Full-compute twice to verify stability; write difference report
                    diffs_csv = os.path.join(out_dirs["runs"], f"consistency_diffs_{time.strftime('%Y%m%d_%H%M%S')}.csv")
                    with open(diffs_csv, "w", encoding="utf-8") as f:
                        f.write("prompt_idx,step,logits_match,hidden_cos_mean\n")
                        for pi, prompt in enumerate(prompts):
                            s1 = engine.encode_prompt(prompt)
                            s2 = engine.encode_prompt(prompt)
                            prev1_h = prev2_h = None
                            prev1_z = prev2_z = None
                            for t in range(num_steps):
                                o1 = engine.step(s1, t, compute_mask=None)
                                o2 = engine.step(s2, t, compute_mask=None)
                                logits_match = int(np.allclose(o1.logits, o2.logits, atol=1e-6))
                                hidden_cos_mean = 1.0
                                if prev1_h is not None and prev2_h is not None:
                                    c1 = cosine_similarity_tokens(o1.hidden_by_layer, prev1_h)
                                    c2 = cosine_similarity_tokens(o2.hidden_by_layer, prev2_h)
                                    hidden_cos_mean = float(0.5 * (np.mean(c1) + np.mean(c2)))
                                f.write(f"{pi},{t},{logits_match},{hidden_cos_mean:.6f}\n")
                                prev1_h = [h.copy() for h in o1.hidden_by_layer]
                                prev2_h = [h.copy() for h in o2.hidden_by_layer]
                                prev1_z = o1.logits.copy()
                                prev2_z = o2.logits.copy()
            elif mode == "rule_gate":
                stepwise_thresholds_obj: Optional[StepwiseThresholds] = None
                if quantiles_path:
                    if not os.path.exists(quantiles_path):
                        raise FileNotFoundError(f"Quantiles file not found: {quantiles_path}")
                    if cos_quantile_level is None or delta_quantile_level is None:
                        raise ValueError("Stepwise thresholds require one_minus_cos_quantile and delta_l2_quantile")
                    table = StepQuantileTable.load(quantiles_path)
                    # Robust handling: quantile tables often start at step=1 since step=0 has no previous state.
                    # Build thresholds for steps [0..num_steps-1] by copying from the nearest available step when missing.
                    step_keys_present = sorted(int(k) for k in table.steps.keys())
                    if not step_keys_present:
                        raise ValueError(f"No steps present in quantiles file: {quantiles_path}")
                    max_available = max(step_keys_present)
                    step_count = min(num_steps, int(table.num_steps) if table.num_steps is not None else (max_available + 1))

                    def _nearest_step(s: int) -> int:
                        if s in step_keys_present:
                            return s
                        # choose nearest lower step; if none, use smallest present
                        lower = [k for k in step_keys_present if k <= s]
                        if lower:
                            return max(lower)
                        return step_keys_present[0]

                    raw_one_minus_cos = [table.value("one_minus_cos", _nearest_step(step), cos_quantile_level) for step in range(step_count)]
                    raw_delta = [table.value("delta_l2_norm", _nearest_step(step), delta_quantile_level) for step in range(step_count)]
                    one_minus_cos_vals = clip_sequence(raw_one_minus_cos, clip_min, clip_max)
                    delta_vals = clip_sequence(raw_delta, clip_min, clip_max)
                    if stepwise_ema is not None:
                        one_minus_cos_vals = apply_ema(one_minus_cos_vals, stepwise_ema)
                        delta_vals = apply_ema(delta_vals, stepwise_ema)
                    cosine_tau_steps = [float(np.clip(1.0 - val, -1.0, 1.0)) for val in one_minus_cos_vals]
                    delta_l2_steps = [float(max(val, 0.0)) for val in delta_vals]
                    gamma_steps = [float(gamma_margin)] * step_count
                    stepwise_thresholds_obj = StepwiseThresholds(
                        cosine_tau=cosine_tau_steps,
                        delta_l2_rho=delta_l2_steps,
                        gamma_margin=gamma_steps,
                        source_path=quantiles_path,
                    )
                    if kl_quantile_level is not None:
                        kl_values = [table.value("kl", _nearest_step(step), kl_quantile_level) for step in range(step_count)]
                        kl_values = clip_sequence(kl_values, clip_min, clip_max)
                        if stepwise_ema is not None:
                            kl_values = apply_ema(kl_values, stepwise_ema)
                        if kl_values:
                            watchdog_max_kl = max(float(max(kl_values)), watchdog_max_kl)
                            resolved_manifest["rule_gate"]["watchdog_max_kl"] = watchdog_max_kl
                    print(f"[rule_gate] Loaded stepwise thresholds from {quantiles_path} (steps={step_count})")

                rule_cfg = RuleGateConfig(
                    cosine_tau=tau,
                    delta_l2_rho=rho,
                    consecutive_m=consecutive_m,
                    freeze_K=freeze_K,
                    watchdog_min_cos=watchdog_min_cos,
                    watchdog_max_kl=watchdog_max_kl,
                    layer_recompute_M=layer_recompute_M,
                    gamma_margin=gamma_margin,
                )
                run_rule_gate(
                    engine,
                    prompts,
                    out_dirs,
                    num_steps,
                    rule_cfg,
                    risk_delta=risk_delta,
                    budget_fraction=budget_fraction,
                    consistency_check=args.consistency_check,
                    stepwise_thresholds=stepwise_thresholds_obj,
                    calibrator_cfg=calibrator_cfg,
                    decision_logger=decision_logger,
                    profile=args.profile if hasattr(args, "profile") else None,
                    labels=prompt_labels,
                    task_name=task,
                    gen_max_new=max_new_tokens,
                )
            elif mode == "learned_gate":
                default_gate_path = str(Path(models_root) / "gates" / "learned_gate_latest.npz") if models_root else None
                weights_path = coalesce(args.learned_gate_weights, cfg.get("learned_gate_weights"), default_gate_path)
                if not weights_path or not os.path.exists(weights_path):
                    raise FileNotFoundError(f"Learned gate weights not found at {weights_path}")
                weights = LearnedGateWeights.from_npz(weights_path)
                threshold_override = coalesce(args.learned_gate_threshold, cfg.get("learned_gate_threshold"))
                if threshold_override is not None:
                    weights.threshold = float(threshold_override)
                learned_gate_cfg = LearnedGateConfig(
                    freeze_K=int(coalesce(args.learned_gate_freeze_K, cfg.get("learned_gate_freeze_K"), freeze_K)),
                    min_consecutive=int(coalesce(args.learned_gate_min_consecutive, cfg.get("learned_gate_min_consecutive"), consecutive_m)),
                )
                gate = LearnedGate(weights, learned_gate_cfg)
                run_learned_gate(
                    engine,
                    prompts,
                    out_dirs,
                    num_steps,
                    gate,
                    watchdog_min_cos=watchdog_min_cos,
                    watchdog_max_kl=watchdog_max_kl,
                    risk_delta=risk_delta,
                    budget_fraction=budget_fraction,
                    consistency_check=args.consistency_check,
                    calibrator_cfg=calibrator_cfg,
                    decision_logger=decision_logger,
                    profile=args.profile if hasattr(args, "profile") else None,
                    labels=prompt_labels,
                    task_name=task,
                    gen_max_new=max_new_tokens,
                )
            elif mode == "adaptive":
                scheduler_cfg = AdaptiveSchedulerConfig(
                    lte_eps=float(args.lte_eps),
                    min_consecutive=int(args.lte_min_consec),
                    max_stride=int(args.max_stride),
                    base_stride=1,
                )
                run_adaptive(
                    engine,
                    prompts,
                    out_dirs,
                    num_steps,
                    scheduler_cfg=scheduler_cfg,
                    risk_delta=risk_delta,
                    skip_budget=float(args.adaptive_budget),
                    hard_budget=bool(args.adaptive_hard_budget),
                    labels=prompt_labels,
                    task_name=task,
                    gen_max_new=max_new_tokens,
                    decision_logger=decision_logger,
                )
            elif mode == "oracle":
                teacher_artifacts = run_teacher(
                    engine,
                    prompts,
                    out_dirs,
                    num_steps,
                    dump_features_dir=dump_features_dir,
                    label_thresholds=label_thresholds,
                    oracle_eval=True,
                    task_name=task,
                    exp_name=exp_name,
                    labels=prompt_labels,
                )
            elif mode == "baselines":
                run_baselines(out_dirs, num_steps)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
    finally:
        close_loggers(decision_loggers)

    if teacher_artifacts:
        if teacher_artifacts.features_path:
            print(f"[teacher] token features saved to {teacher_artifacts.features_path}")
        if teacher_artifacts.quantiles_path:
            print(f"[teacher] quantiles saved to {teacher_artifacts.quantiles_path}")
        if teacher_artifacts.outputs_path:
            print(f"[teacher] final outputs saved to {teacher_artifacts.outputs_path}")

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
