# Delta-Compute Gating for Diffusion LLM — Master Plan (v3, 2025-10-03)

> One-line goal. Reduce end-to-end dLLM latency and FLOPs by computing only the necessary deltas across diffusion steps — via rule-based freezing (P2), a lightweight learned gate (P3), and adaptive step skipping (P4) — while formally controlling risk and preserving task quality.

---

## 0) TL;DR (delta vs v2)
- Corrected metrics and math: formal definitions for δ_frozen, coverage, true FLOPs, Compute-AUC, and skip-regret; unified latency/GPU-util protocol.
- Real speed: engines must skip matmuls (sub-batch/block-sparse). Add micro-perf curve and per-layer cost model.
- Step-wise thresholds: P2/P3 use per-step quantiles of (1−cos), ΔL2_norm, KL logged in P1; conformal risk is per-step with smoothing/clip.
- Trustworthy risk: report δ_frozen only over actually skipped/frozen positions; KL hard guard; rollback budget ≤3%.
- Adaptive limits: P4 uses normalized LTE + a dual (risk) condition, step-wise max stride, tail-guard, and β_steps cap.
- Oracle/baseline sandwich: always show Compute-AUC and Oracle→Gate gap@Quality; must beat fixed-stride/index-only/layer-throttle baselines.

---

## 1) Scope & Non-Goals
- In scope: inference-time only; per-token/per-layer freezing and per-sequence step skipping for locally loaded diffusion LLMs.
- Not in v3: backbone training; heavy distillation; prod hardening beyond a reproducible research prototype.

---

## 2) Success Criteria & Formal Metrics
- Latency (ms/seq, batch=1): target ≥1.5× vs Teacher; report mean/p50/p90.
- Throughput (seq/s @ batch 4–8).
- True FLOPs / MMA counts (prefer profiler; else cost model).
- Compute-AUC: Quality vs %FLOPs (Teacher=100%), 95% bootstrap CI.
- Skip-regret: at fixed quality Q*, min FLOPs% − 100%.
- Quality deltas with 95% CI: Wikitext-2 PPL ≤ +3%; LAMBADA-open Acc ≥ −1pp; Tiny GSM8K EM ≥ −1pp.
- Shape preservation: final token consistency (mean) + coverage; optional seq EM and final logit KL (goal ≤0.02).
- δ_frozen (core risk): false-positive rate over frozen/skipped items only. Target ≤1% (dataset-dependent).
- Latency/GPU-util protocol: warmup once; fixed decoding; NVML/dmon @200–500ms for gpu_util_gpu_mean and gpu_util_mem_mean; exclude heavy logging from timing.

---

## 3) Environment
```bash
conda create -y -n dcllm python=3.10
conda activate dcllm
pip install -r requirements.txt  # PyTorch ≥ 2.2, match CUDA
```
```python
import torch, random, numpy as np
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(False)
seed = 42
torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
```

---

## 4) Repository Layout
Full local-first layout (v2) with v3 notes:

```
dcllm/
  README.md
  master_plan.md                      # this file

  configs/
    tasks.yaml                        # eval suites & dataset sizes (see §6)
    thresholds.yaml                   # P2 profiles per dataset (see §8)
    paths.yaml                        # local data/model/output roots

  scripts/
    run_teacher_local.sh              # P1
    run_rule_gate_local.sh            # P2
    run_learned_gate_local.sh         # P3
    run_adaptive_steps_local.sh       # P4
    make_report.sh

  src/
    run.py                            # entry: teacher | rule_gate | learned_gate | adaptive | oracle | baselines
    engine/
      local_engine.py                 # local model loader + step() for diffusion model
      cache.py                        # reuse buffers, pinned tensors, mask routing
    probes/
      metrics.py                      # cosine, ΔL2, KL, entropy, margin, centrality
      sampler.py                      # KL/entropy moving averages, step diagnostics
    gating/
      rule_gate.py                    # P2: token/layer freeze, watchdog
      learned_gate.py                 # P3: TorchScript logistic/MLP gate
      scheduler.py                    # P4: adaptive step skipping (stride control)
      calibrate.py                    # ROC/PR/threshold search + per-dataset profiles
    eval/
      tasks.py                        # loaders: Wikitext-2, LAMBADA-open, Tiny GSM8K
      scorer.py                       # PPL/Acc/EM; CI & bootstrap utilities
      consistency.py                  # token/seq consistency, final logit KL
    viz/
      plots.py tables.py failure_gallery.py
    utils/
      logging.py timer.py profile.py io.py seed.py

  reports/
    runs/
    figures/
    artifacts/
      teacher_outputs/
      features/
      gates/
```

v3 adds: per-step conformal risk logging, normalized LTE traces, micro-perf sweep, and δ_frozen/coverage reporting.

---

## 5) Data & Evaluation (Tiering)
- Tier A — Smoke (fast, statistically usable)
  - Wikitext-2 (PPL): ~2,000 segments
  - LAMBADA-open (Acc): 2,000 samples (SE≈1.1pp @ p≈0.6)
  - Tiny GSM8K (EM): 1,000–2,000 samples
- Tier B — Day-scale: CNN/DM-mini, MMLU-mini (5×100), long-context scratchpads.
- Tier C — Stretch: GSM8K full, larger MMLU, rare-word LAMBADA, cross-backbone.

Deterministic subsampling via configs/tasks.yaml.

---

## 6) Config Examples (v3-aligned)
- configs/tasks.yaml (smoke triplet): set n_eval to 2000/2000/1500 for Tier A. Alternatively override with `--task` + `--num_prompts` at runtime.
- configs/thresholds.yaml (P2): include dataset-specific risk deltas and step-wise quantiles; use EMA smoothing and [p10,p99] clipping.
- configs/adaptive.yaml (P4): normalized LTE, min_consecutive, max_stride_step (tail-guard), β_steps.

Concrete YAMLs (from v2, keep for completeness):

`configs/tasks.yaml`
```yaml
smoke:
  max_new_tokens: 128
  datasets:
    - name: wikitext
      subset: wikitext-2-raw-v1
      n_eval: 2000
      metric: ppl
    - name: lambada_openai
      subset: en
      n_eval: 2000
      metric: acc
    - name: gsm8k
      subset: main
      n_eval: 1500
      metric: em
```

`configs/paths.yaml`
```yaml
data_root: /path/to/data
models_root: /path/to/models
outputs_root: /path/to/outputs
```

`configs/thresholds.yaml` (dataset profiles—use as defaults; for per-step quantiles and δ see §8)
```yaml
rule_gate:
  watchdog: {min_cos: 0.90, max_kl: 0.02}

  wikitext2:
    conservative: {tau: 0.97, rho: 0.05, m: 2, K: 1, M: 2, gamma_margin: 0.10}
    balanced:    {tau: 0.95, rho: 0.10, m: 2, K: 2, M: 2, gamma_margin: 0.08}
    aggressive:  {tau: 0.90, rho: 0.15, m: 1, K: 3, M: 3, gamma_margin: 0.05}

  lambada_open:
    conservative: {tau: 0.97, rho: 0.05, m: 2, K: 1, M: 2, gamma_margin: 0.12}
    balanced:    {tau: 0.95, rho: 0.10, m: 2, K: 2, M: 2, gamma_margin: 0.10}
    aggressive:  {tau: 0.90, rho: 0.20, m: 1, K: 3, M: 3, gamma_margin: 0.08}

  gsm8k_tiny:
    conservative: {tau: 0.97, rho: 0.05, m: 2, K: 1, M: 1, gamma_margin: 0.15}
    balanced:    {tau: 0.95, rho: 0.10, m: 2, K: 2, M: 1, gamma_margin: 0.12}
    aggressive:  {tau: 0.90, rho: 0.20, m: 1, K: 2, M: 1, gamma_margin: 0.10}
```

---

## 7) P1 — Teacher Traces & Visuals
Purpose: full-compute run; log per-step/layer/token features to expose stability and cost hotspots; write teacher outputs for consistency.

Log (per t, L, i): cosine, ΔL2_norm, KL, entropy, margin, σ-normalized deltas, step_frac; attention centrality/entropy; LTE probe (Euler/Heun residual). Persist per-step quantiles q_p^(t) for (1−cos), ΔL2_norm, KL.

Persist: teacher outputs, features NPZ (for P3), quantile tables, profiler summaries.

Visuals: heatmaps (layer×step), hist/violin, layer-time breakdown, token stability spans, latency tails.

Correct consistency metrics (v2 details, retained):
- Token-level final consistency (per prompt): fraction of positions whose final generated token matches Teacher; averaged to final_token_consistency_mean.
- Sequence exact match (optional) and final logit KL mean.
- If Teacher outputs are missing, do not write -1.0; log missing_teacher_output=true and include coverage in summary.

What to keep (evaluation):
- Latency mean/p50/p90, throughput, true FLOPs rel, GPU util.
- Quality baseline metrics with 95% CI.
- Final token consistency mean + coverage; optional final logit KL.

Five detailed samples (trace):
- Run a trace pass with `--num_prompts 5 --consistency_full_compute` to produce per-step stability CSVs and a consistency diffs CSV. Archive all outputs under reports/artifacts/traces/P1/{dataset}/.

---

## 8) P2 — Rule-Based Gating (must save real compute)
- Sub-batch/block-sparse only; never compute-then-overwrite.
- Step-wise thresholds from P1 q_p^(t):
  - (1−cos) ≤ q_0.88–0.90^(t), ΔL2_norm ≤ q_0.30–0.50^(t), margin ≥ γ (0.08–0.10).
- Conformal risk (per-step): s=max{ΔL2_norm, 1−cos, KL}; compute q_δ^(t) on calibration; smooth (EMA 0.9) and clip [p10,p99].
- Watchdog + rollback: KL>0.003 or (1−cos) spike → rollback and recompute; rollback_time_ratio ≤3%.
- m/K/M and heavy layers: early m=2,K=1; mid m=1,K=2; late m=1,K=2–3; throttle only costliest layers; add K±1 jitter.
- Optional budget β (default 0.60): greedy knapsack on benefit/risk.

Additional v2 details (kept):
- Attention-graph-aware freezing: cluster tokens via attention centrality/community; freeze clusters jointly; keep anchors live. Provide community-level skip tables and failure-gallery heatmaps.
- Budget-aware controller: accept β as fraction of Teacher FLOPs; score candidate skips by expected_FLOPs_saved/risk and select greedily.

Outputs to keep:
- Skip ratios (token×layer×step), latency p50/p90, throughput, true FLOPs rel.
- δ_frozen + coverage, consistency, rollback stats.
- Micro-perf: unfrozen_ratio → step() latency curve; mark non-linearity threshold.

Five detailed samples (trace):
- Add `--num_prompts 5 --consistency_check` to emit per-prompt skip_stats_*.csv and a consistency comparison against a full-compute baseline. Archive under reports/artifacts/traces/P2/{dataset}/.

---

## 9) P3 — Lightweight Learned Gate
- Supervision (strict): final argmax unchanged, final-step margin ≥0.07, watchdog-safe within K; ambiguous → drop.
- Features: [cos_norm, ΔL2_norm, KL, entropy, margin, attn_sum, attn_entropy, σ_t, step_frac, layer_band, community_id, risk_score, budget_remaining, step_mod].
- Model: logistic/2-layer MLP (<50k params), TorchScript. Cost-weighted positives ∝ FLOPs_saved; handle imbalance.
- Calibrate: ROC/PR; temperature or isotonic. Conformal δ as in P2; optional sequential SPRT (α=β=0.005).
- Deployment: vectorized features on GPU; output (p_freeze, K_len∈{1,2,3}, risk); keep watchdog.

Training/calibration details (v2 retained, reconciled):
- Split by prompts; early stop on AUROC and monitor risk-calibrated precision.
- Handle class imbalance (focal or pos_weight); sweep thresholds to populate Compute-AUC.
- Conformal δ matching §8; optional sequential SPRT α=β=0.005.

Operating targets (dataset):
- Wikitext-2: precision ≥0.98, skip 10–15%, δ_frozen ≤0.5–1.0%.
- LAMBADA-open: precision ≥0.995, skip 8–12%, δ ≤0.75%.
- Tiny GSM8K: precision ≥0.995, skip 5–10%, δ ≤1.0%.

Five detailed samples (trace):
- Run eval with `--num_prompts 5 --consistency_check` to produce per-prompt skip_stats_*.csv and consistency vs Teacher; archive traces under reports/artifacts/traces/P3/{dataset}/.

---

## 10) P4 — Adaptive Step Scheduling
- Signals: normalized LTE (Euler vs Heun), risk dual condition, stride caps (step-wise), tail-guard, β_steps cap.
- Defaults example: use_normalized_lte=true; lte_eps_step≈[0.020..0.012]; min_consecutive≈[4..2]; max_stride_step≈[1..4→2 tail]; beta_steps=0.20; kl_guard=0.003.

Outputs to keep:
- Skip ratio estimate per sequence; latency p50/p90; LTE/risk/stride traces; risk thresholds; scheduler settings. Compute-AUC contribution.

Five detailed samples (trace):
- Add a 5-prompt run and archive the rendered LTE/risk/stride traces and summary under reports/artifacts/traces/P4/{dataset}/.

---

## 11) P5 — Oracle & Baselines
- Oracle headroom and baseline floor: fixed-stride (t0,s), index-only, layer-throttle.
- Report: Pareto (latency vs quality), Compute-AUC, skip-regret, gap@Quality.

Five detailed samples (trace):
- For each dataset, keep the top-5 cases with largest oracle gap and archive the inputs/outputs and summary rows.

---

## 12) Run Tier A Now (end-to-end instructions)

Common settings
- Engine: d2f unless noted; use_d2f_lora=false initially.
- Steps/tokens: num_steps=12, max_new_tokens=128 (match P1/P2 configs).
- Outputs: reports/ (auto-structured by phase/mode/engine/exp_name).

Dataset sizes (Tier A)
- Wikitext-2: --task wikitext --num_prompts 2000
- LAMBADA-open: --task lambada --num_prompts 2000
- Tiny GSM8K: --task gsm8k --num_prompts 1500 (or 2000 if time permits)

Run commands (local, example)
- P1 Teacher (full run + 5-trace):
  - Full: python -m src.run --mode teacher --config configs/p1_smoke.yaml --task wikitext --num_prompts 2000 --oracle_eval --lte_probe heun --dump_features_dir reports/artifacts/features
  - Trace 5: python -m src.run --mode teacher --config configs/p1_smoke.yaml --task wikitext --num_prompts 5 --consistency_full_compute --exp_name wikitext_trace5
  - Repeat for lambada (2000) and gsm8k (1500).
- P2 Rule gate (risk + budget):
  - Wikitext: python -m src.run --mode rule_gate --config configs/p2_smoke.yaml --task wikitext --num_prompts 2000 --profile balanced --risk_delta 0.005 --budget_fraction 0.60
  - Trace 5: add --num_prompts 5 --consistency_check --exp_name wikitext_rule_trace5
  - LAMBADA: use --profile conservative --risk_delta 0.0075; GSM8K: balanced, 0.010.
- P3 Learned gate (train+eval):
  - Train from latest P1 features: scripts/sbatch_p3_learned_gate.sbatch (auto-discovery) or python -m src.learned.train_gate --features reports/artifacts/features/teacher_features_*.npz --out models/gates/learned_gate_latest.npz
  - Eval: python -m src.run --mode learned_gate --config configs/p3_smoke.yaml --task wikitext --num_prompts 2000 --learned_gate_weights models/gates/learned_gate_latest.npz
  - Trace 5: add --num_prompts 5 --consistency_check --exp_name wikitext_learned_trace5
- P4 Adaptive:
  - python -m src.run --mode adaptive --config configs/p4_smoke.yaml --task wikitext --num_prompts 2000 --lte_probe heun --lte_eps 0.015 --lte_min_consec 2 --max_stride 4 --adaptive_budget 0.20
  - Trace 5: add --num_prompts 5 --exp_name wikitext_adapt_trace5
- P5 Oracle & baselines:
  - Oracle: scripts/sbatch_p5_oracle.sbatch with CONFIG_PATH=configs/p5_smoke.yaml and per-dataset overrides (see wrappers below).
  - Baselines: scripts/sbatch_p5_baselines.sbatch.

What to save per run
- Quality: PPL/Acc/EM + 95% CI, consistency, final logit KL (opt).
- Risk: δ_frozen + coverage; risk threshold history; rollback stats.
- Performance: latency mean/p50/p90, throughput, GPU util, true FLOPs rel.
- Visuals: heatmaps/histograms; Pareto; Compute-AUC; skip-regret; failure gallery.

Trace 5 per dataset per phase
- Keep 5 per-prompt detailed artifacts: for P1/P2/P3, capture skip_stats_*.csv and consistency diffs (where applicable). For P4, archive LTE/risk/stride traces and summary. Store under reports/artifacts/traces/P{phase}/{dataset}/ with exp_name suffix *_trace5.

---

## 13) One-Click sbatch (Tier A triplet)
The repository now includes wrappers that launch Wikitext-2, LAMBADA-open, and Tiny GSM8K for each phase (P1–P5), plus an additional 5-sample trace run per dataset. See scripts:
- scripts/sbatch_p1_smoke_all.sbatch
- scripts/sbatch_p2_smoke_all.sbatch
- scripts/sbatch_p3_smoke_all.sbatch
- scripts/sbatch_p4_smoke_all.sbatch
- scripts/sbatch_p5_smoke_all.sbatch

Usage examples
- sbatch scripts/sbatch_p1_smoke_all.sbatch
- sbatch scripts/sbatch_p2_smoke_all.sbatch
- sbatch scripts/sbatch_p3_smoke_all.sbatch
- sbatch scripts/sbatch_p4_smoke_all.sbatch
- sbatch scripts/sbatch_p5_smoke_all.sbatch

Each wrapper:
- Activates conda env `dcllm` and sets CUDA module if available.
- Runs full Tier A sizes and then a 5-sample trace job per dataset with appropriate exp_name suffix.
- For P2/P3, sets dataset-appropriate `--profile` and `--risk_delta` (Wikitext 0.005; LAMBADA 0.0075; GSM8K 0.010) and budget 0.60.

---

## 14) Reporting & Visualization (minimum set)
- δ_frozen–Coverage curves (overall & per-step); choose operating points.
- Skip (token/step) vs latency p50/p90 scatter; expose non-linearity threshold.
- Stride trace overlaid with LTE_norm & risk (verify triggers).
- Per-layer time share (Teacher vs Gate) to prove costly layers throttled.
- Layer×step heatmaps; hist/violin; ROC/PR & calibration (P3).
- Compute-AUC & skip-regret with 95% CI; failure gallery.

Known gaps fixed (from v2; keep for tracking):
- final_token_consistency now correct and coverage recorded.
- GPU util via NVML/dmon; gpu_util_gpu_mean, gpu_util_mem_mean saved.
- Masking without speed disallowed; require sub-batching/layer throttling validated by profiler and micro-perf suite.
- Sample sizes increased; defaults and CLI override documented.
- Overhead budget: features + gating <3%.
- Oracle/baseline sandwich codified; Compute-AUC & skip-regret required.
- LTE-controlled adaptive schedule with rollback; conformal δ-bounded gating and optional SPRT; σ_t normalization for drift.

---

## 15) Risks & Guardrails
- Accumulated error → watchdog+rollback; cap per-step frozen proportion.
- Attention coupling → community-aware freezing; keep high-centrality anchors live.
- Distribution shift → stress suites + sliding-window recalibration of q_δ^(t).
- Risk drift → alert if δ_frozen exceeds target for 3 consecutive windows.
- Overhead creep → features+gating <3% runtime; micro-perf thresholding.
- Budget misuse → never exceed FLOPs budget silently; Quality@Budget watchdog.

Deliverables (v2, kept):
- Code: delta-freeze, adaptive-steps, router, oracle-eval, baseline-eval, conformal calibration scripts, microbench tooling, budget controller.
- Metrics: Compute-AUC, skip-regret tables, δ guarantees, true FLOPs traces, LTE histograms.
- Artifacts: Oracle/baseline Pareto plots, Quality@Budget curves, OOD breakdowns, microbench sweeps, budget logs.
- Cross-model evidence across ≥2 dLLM architectures and decoding settings.
- W&B dashboard and reproducible report via scripts/make_report.sh.

---

## 16) Milestones
- P0 (Day 0–0.5) Env & skeleton; engine compiles.
- P1 (0.5–1) Teacher traces + quantiles + figures.
- P2 (1–2) Rule gate + step-wise thresholds + δ_frozen reporting.
- P3 (2–3) Learned gate + cost-weighted training + calibration.
- P4 (3–4) Adaptive (normalized LTE, dual condition, tail-guard, β_steps).
- P5 (4–5) Oracle/Baseline, Compute-AUC, gap@Quality; combine + ablations.

---

## Appendix A — Cost Model (retained)
For a transformer block with L layers, sequence length S, heads H, key dim d_k, model dim d, FFN hidden d_ff:
FLOPs_attn ≈ 2L·H·d_k·S²; FLOPs_ffn ≈ 2L·S·d·d_ff. Use measured per-layer shares to weight P2/P3 and compute expected FLOPs_saved.

## Appendix B — Pseudocode (v2 + v3)
- P2 Rule gate and P3 Learned gate pseudocode retained from v2 and v3 drafts; includes rollback buffer usage, conformal thresholds, budget controller selection, and stride schedule updates.

## Appendix C — Glossary (retained)
- cosine: cosine similarity between hidden at steps t and t−1.
- ΔL2 ratio: ||h_t − h_{t-1}|| / (||h_{t-1}|| + ε).
- KL: KL(softmax z_t || softmax z_{t-1}).
- entropy, margin: uncertainty & top-1 confidence gap.
- attention centrality: total attention mass received by a token.

