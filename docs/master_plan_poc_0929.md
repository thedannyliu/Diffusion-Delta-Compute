````markdown
# Delta-Compute Gating for Diffusion LLM — **Master Plan (v2)**

> **Goal (1-liner).** Cut end-to-end dLLM **latency & throughput** by computing **only necessary deltas** across diffusion steps—via **rule-based freezing**, **lightweight learned gating**, and **adaptive step skipping**—while **strictly controlling task quality**.

---

## 0. TL;DR
- **Hypothesis.** Consecutive diffusion steps change a subset of tokens/layers only slightly. If we **freeze+reuse** stable parts and **skip steps** when sequence-level change is low, we save FLOPs with minimal quality loss.
- **Phases.** P0 Env → **P1 Teacher traces** → **P2 Rule gate** → **P3 Learned gate** → **P4 Adaptive step scheduling** → **P5 Combine + ablations + report**.
- **Primary KPI.** ≥ **1.5× latency speedup** (batch=1) at ≤ **0.2σ** drop vs Teacher (or: **Wikitext-2 PPL ≤ +3%**, **LAMBADA Acc ≥ −1pp**, **GSM8K EM ≥ −1pp**). Also report a **2×** speed tier with controlled degradation.

---

## 1. Scope & Non-Goals
**In scope**
- **Inference-time only** (no backbone retraining).
- **Per-token/per-layer** delta-compute (freeze/copy) and **step skipping**.
- Works with **locally loaded** diffusion LLMs (no cluster dependence).

**Out of scope (v1)**
- Training new diffusion LLMs.
- Heavy distillation/fine-tuning of the backbone.
- Productionization; we ship a **research-grade**, reproducible prototype.

---

## 2. Success Criteria, KPIs & Statistical Reporting
**Speed**
- **Latency** (ms/sequence, batch=1): target ≥ **1.5×** vs Teacher.
- **Throughput** (seq/s @ batch=4–8).

**Quality**
- **Wikitext-2 PPL**: Δ% = `(PPL_accel − PPL_teacher) / PPL_teacher × 100%` ≤ **+3%**.
- **LAMBADA-open Accuracy**: Δpp ≥ **−1pp** (absolute percentage-point).
- **Tiny GSM8K EM**: Δpp ≥ **−1pp**.

**Teacher similarity (shape preservation)**
- **Final token consistency (mean)**: fraction of positions whose final **generated token** matches Teacher. **Must** be reported.
- (Optional) **Final logit KL (mean)** at last step; aim ≤ **0.02**.
- (Optional) **Sequence exact match rate** (whole output string equality).

**Statistical reporting**
- For **Acc/EM**: 95% **binomial CI**.
- For **PPL**: bootstrap 95% CI over per-token NLL.
- Always report **p50/p90 latency**, and plot **Pareto (Latency vs Quality)** with CIs.

---

## 3. Environment Setup (Local)
```bash
conda create -y -n dcllm python=3.10
conda activate dcllm
pip install -r requirements.txt
# Optional GPU: match torch/cu to your system (PyTorch ≥ 2.2)
````

**Determinism (for eval)**

```python
import torch, random, numpy as np
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(False)
seed = 42
torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
```

---

## 4. Repository Layout (v2, local-first)

```
dcllm/
  README.md
  master_plan.md                      # this file

  configs/
    tasks.yaml                        # eval suites & dataset sizes (see §6)
    thresholds.yaml                   # P2 profiles per dataset (see §8.3)
    paths.yaml                        # local data/model/output roots

  scripts/
    run_teacher_local.sh              # P1
    run_rule_gate_local.sh            # P2
    run_learned_gate_local.sh         # P3
    run_adaptive_steps_local.sh       # P4
    make_report.sh

  src/
    run.py                            # entry: teacher | rule_gate | learned_gate | adaptive
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
      plots.py                        # heatmaps, histograms, ROC/PR, Pareto, latency tails
      tables.py                       # CSV writers (per-run summaries)
      failure_gallery.py              # attach attention overlays & diffs
    utils/
      logging.py timer.py profile.py  # timers, nvml/dmon sampling, JSONL writers
      io.py seed.py

  reports/
    runs/                             # JSONL/CSV summaries
    figures/                          # auto-saved figures
    artifacts/
      teacher_outputs/                # per-prompt Teacher final outputs
      features/                       # P1 token features for P3 training
      gates/                          # serialized learned gates (TorchScript)
```

> **Why this layout?**
> • Removes cluster coupling. • Makes **consistency** and **profiling** first-class modules. • Ensures **Teacher outputs** are persisted for proper comparisons. • Keeps **calibration** (threshold search) reproducible.

---

## 5. Data & Evaluation Tasks (with sample sizes)

**Tier A — Smoke (fast, statistically usable)**

* **Wikitext-2** (PPL): ~**2k segments** (good bootstrap CIs)
* **LAMBADA-open** (Acc): **2,000** samples (SE≈1.1pp @ p≈0.6)
* **Tiny GSM8K** (EM): **1,000–2,000** samples

**Tier B — Day-scale**

* **CNN/DM-mini** (ROUGE-L)
* **MMLU-mini** (5 subjects × 100 qs)

**Tier C — Stretch**

* **GSM8K full dev**, larger MMLU slice

All datasets via `datasets`; deterministic subsampling in `configs/tasks.yaml`.

---

## 6. Config Examples

**`configs/tasks.yaml`**

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

**`configs/paths.yaml`**

```yaml
data_root: /path/to/data
models_root: /path/to/models
outputs_root: /path/to/outputs
```

---

## 7. P1 — Teacher Traces & Visualizations (Day 0–1)

**Purpose.** Run full-compute Teacher (no skipping) and log **per-step, per-layer** features to find stability & cost hotspots.

### 7.1 Logged features (per step `t`, layer `L`, token `i`)

* **Cosine**(hₜ, hₜ₋₁), **ΔL2 ratio** ‖hₜ−hₜ₋₁‖ / (‖hₜ₋₁‖+ε)
* **Logit KL** KL(softmax zₜ || softmax zₜ₋₁) per token
* **Entropy**, **margin** = p(top1) − p(top2)
* **Attention centrality** (sum of incoming attention); **attention entropy**
* **Per-layer timing** (timers or `torch.profiler` on a small subset)

### 7.2 Persisted artifacts

* `reports/artifacts/teacher_outputs/*.jsonl` → **final outputs** per prompt
* `reports/artifacts/features/teacher_features_*.npz` → token features (for P3)
* `reports/runs/teacher_summary.json` → latency/throughput/p50/p90, GPU mem, CIs

### 7.3 **Correct** consistency metrics (fix for `-1.0`)

* **Token-level final consistency** (per prompt):
  `consistency = (# positions with generated token == Teacher token) / length`
  Average across prompts → `final_token_consistency_mean`.
* **Sequence exact match**: whole string equality (0/1), averaged.
* (Optional) **Final logit KL** mean across positions.

> If Teacher outputs are missing, **do not** write `-1.0`; log `"missing_teacher_output": true` and skip that prompt. Include **coverage** (fraction of prompts with valid Teacher comparisons) in the summary.

### 7.4 Visualizations (richer set)

1. **Layer×Step heatmaps**: cosine (mean ± IQR), ΔL2, logit-KL
2. **Histograms & violin plots**: cosine / ΔL2 across all (L, t, i)
3. **Token stability lengths**: distribution of **max consecutive stable steps** (by τ/ρ & margin γ)
4. **Per-layer time breakdown**: stacked bars of attention/FFN/other
5. **Centrality vs “freeze probability”** scatter (overlay P2/P3 decisions later)
6. **Latency tails**: p50/p90 density plots per mode (Teacher later vs P2/P3)

---

## 8. P2 — **Rule-Based Gating** (Baseline, pre-configured)

**Idea.** Freeze tokens (and optionally shallow layers) that look **stable**; **watchdog** prevents drift.

### 8.1 Algorithm (per step)

1. Compute features for **unfrozen** tokens (cos, ΔL2, KL, entropy, margin).
2. Update **stability counters**; if `(cos ≥ τ) OR (ΔL2 ≤ ρ)` for **m** consecutive steps, set **freeze window** of **K** steps.
3. Build `token_mask` and optional **`layer_mask`** for shallow layers (see 8.2).
4. **Watchdog** each step: if `(cos < min_cos)` OR `(KL > max_kl)` for any token in window, **unfreeze & recompute** immediately.

### 8.2 **Layer throttling** (high ROI)

* Identify **heavy layers** from P1 timing; for **L1–L3 (or chosen heavy layers)**, recompute every **M** steps.
* Ensure frozen tokens/layers **skip matmuls** (sub-batching/block-sparse), not “compute then overwrite.”

### 8.3 **Pre-configured threshold profiles (per dataset)**

**`configs/thresholds.yaml`**

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

**Recommended defaults**

* **Wikitext-2** → **Balanced**
* **LAMBADA** → **Conservative**
* **GSM8K Tiny** → **Conservative/Balanced**

### 8.4 Outputs & checks

* **Skip ratio** (token×layer×step), **latency/throughput**, **p50/p90**
* **Quality** (PPL/Acc/EM) with **CIs**
* **Consistency** (token & sequence), **final logit KL (optional)**
* **Unfreeze rate** (watchdog)
* **Failure gallery** with attention overlays

---

## 9. P3 — **Lightweight Learned Gate**

**Supervision.** Label (t, i) `safe_to_freeze` if freezing for next **K** steps **doesn’t change task success** or final logits **argmax/score crossing**.

**Features.** `[cos, ΔL2, KL, entropy, margin, attn_sum, attn_entropy, step_index/num_steps, layer_band]`.

**Model.** Logistic or 2-layer MLP (hidden=64), **< 50k params**, **TorchScript**.

**Training.**

* Split by prompts; early stop on **AUROC** (dev).
* Handle imbalance (focal loss or `pos_weight`).
* **Calibrate** threshold via ROC/PR for target **precision** and **skip**.

**Deployment.**

* **Vectorized** feature computation on GPU; batch predictions.
* Output `(p_freeze, freeze_length ∈ {1,2,3})`; retain **watchdog**.

**Calibration targets (per dataset).**

* **Wikitext-2**: precision ≥ 0.98, **skip ≈ 0.45–0.55**
* **LAMBADA**: precision ≥ 0.995, **skip ≈ 0.25–0.40**
* **GSM8K Tiny**: precision ≥ 0.995, **skip ≈ 0.15–0.30**

**Visuals.**

* **ROC/PR curves** with operating point
* **Calibration curves** (pred vs empirical)
* **Per-layer freeze maps**
* **Latency tails** vs P2/P1

---

## 10. P4 — **Adaptive Step Scheduling** (sequence-level)

**Signal.** Moving average of **mean token logit KL** or **entropy slope**.

**Rule.** If signal below threshold for **r** consecutive steps → **increase stride by +1** (skip one denoise step). Else revert to base stride. Cap **total skipped steps ≤ 20%**.

**Defaults.** `r=2`, initial stride=1, cap=20%.

**Outputs.** Effective steps/sequence, Δ latency/throughput, quality deltas, consistency.

---

## 11. P5 — Combine & Evaluate

Compare **Teacher**, **RuleGate**, **LearnedGate**, **Adaptive**, **Combined (P2+P3+P4)** per dataset:

* **Latency** (mean/p50/p90), **Throughput**, **GPU mem**, **skip ratios**
* **Quality** (PPL/Acc/EM + CI)
* **Consistency** (token/seq), (optional) **final logit KL**
* **Pareto fronts** (Latency vs Quality)
* **Ablations**: (i) w/wo attention features; (ii) vary **K/M**; (iii) Adaptive thresholds×stride

---

## 12. Instrumentation & Overhead Budget (Fixes)

**12.1 Ensure masks save FLOPs**

* Avoid “compute then overwrite.” Implement **sub-batching** for unfrozen tokens and **layer throttling** so attention/FFN matmuls are **skipped** for frozen tokens/layers.

**12.2 GPU utilization (previously 0 → **fixed**)**

* Add **NVML** (`pynvml`) sampling thread at 200–500ms to record `gpu_util_gpu_mean`, `gpu_util_mem_mean`.
* Or background `nvidia-smi dmon -s pucvmt -d 1` and parse logs.
* Use `torch.profiler` on small subsets to map per-layer cost.

**12.3 Overhead budget**

* Gating (features + model) **< 3%** runtime.
* Features computed **vectorized on GPU**, no `.item()`/Python loops.
* Reuse buffers (`engine/cache.py`) to avoid host-device thrash.

---

## 13. Reporting & Visualization (richer)

**Auto-generated figures per run**

1. **Layer×Step heatmaps**: cosine, ΔL2, logit-KL (mean ± IQR)
2. **Cosine/ΔL2 histograms** + **violin plots**
3. **Token stability length** distributions
4. **Per-layer time breakdown** (Teacher vs P2/P3)
5. **Freeze maps** (fraction frozen by layer/step)
6. **ROC/PR** (P3) + **calibration curves**
7. **Latency tails** (p50/p90 density) & **skip-ratio CDFs**
8. **Pareto fronts** (Latency vs Quality) with **95% CIs**
9. **Failure case gallery**: diffs in outputs + attention centrality overlays

**Tables (CSV + dashboard)**

* `{latency_mean/p50/p90, throughput, gpu_mem, gpu_util_gpu_mean, skip_ratio, unfreeze_rate, ppl/acc/em + CI, token_consistency, seq_exact_match, logit_kl}`

**Final PDF** (`reports/final_<date>.pdf`)

* Abstract → Method → Experiments → Ablations → Discussion → Limitations → Appendix
* Repro via `scripts/make_report.sh`.

---

## 14. Risks & Guardrails

* **Accumulated error** → watchdog & periodic recompute; cap per-step **frozen proportion**.
* **Attention coupling** → lower τ (or disallow freezing) for high-centrality tokens.
* **Distribution shift** → train learned gate on diverse prompts; include **adversarial long-range** cases.
* **Overhead creep** → continuous profiling; block PRs exceeding **3%** gate overhead.

---

## 15. Milestones & Owners

* **P0 (Day 0–0.5)**: Env + repo skeleton; local engine compiles. *(Danny)*
* **P1 (Day 0.5–1)**: Teacher traces + figures. *(Danny)*
* **P2 (Day 1–2)**: Rule gate + **dataset presets** + curves. *(Danny)*
* **P3 (Day 2–3)**: Learned gate + calibration. *(Danny)*
* **P4 (Day 3–4)**: Adaptive steps. *(Danny)*
* **P5 (Day 4–5)**: Combine + ablations + draft report. *(Danny; review: Celine/Zhenbang/Jai)*

---

## 16. Checklists (updated)

**Before each run**

* [ ] Git commit hash saved
* [ ] Configs frozen (`.yaml` copied into run dir)
* [ ] Seed fixed; deterministic dataset slice
* [ ] **Teacher outputs present** (or logged missing with coverage)
* [ ] W&B run name `{mode}-{model}-{date}-{gitsha}`

**After each run**

* [ ] **Quality metrics with CIs** written
* [ ] **Consistency** metrics computed (coverage reported)
* [ ] Pareto plots updated; artifacts uploaded
* [ ] **Top 3 failure cases** annotated

---

## 17. Example Local Scripts

**`scripts/run_teacher_local.sh`**

```bash
python -m src.run \
  --mode teacher \
  --tasks_cfg configs/tasks.yaml \
  --paths_cfg configs/paths.yaml \
  --out_dir reports
```

**`scripts/run_rule_gate_local.sh`**

```bash
python -m src.run \
  --mode rule_gate \
  --tasks_cfg configs/tasks.yaml \
  --paths_cfg configs/paths.yaml \
  --thresholds_cfg configs/thresholds.yaml \
  --profile balanced \
  --dataset wikitext2 \
  --out_dir reports
```

**`scripts/run_learned_gate_local.sh`**

```bash
python -m src.run \
  --mode learned_gate \
  --tasks_cfg configs/tasks.yaml \
  --paths_cfg configs/paths.yaml \
  --gate_weights reports/artifacts/gates/learned_gate_latest.ts \
  --calibration_target precision=0.98,skip=0.5 \
  --dataset wikitext2 \
  --out_dir reports
```

**`scripts/run_adaptive_steps_local.sh`**

```bash
python -m src.run \
  --mode adaptive \
  --tasks_cfg configs/tasks.yaml \
  --paths_cfg configs/paths.yaml \
  --adaptive_r 2 --adaptive_cap 0.2 \
  --out_dir reports
```

---

## 18. Pseudocode (fixed & clarified)

**P2 Rule-based (token+layer)**

```python
state = engine.encode(prompt)
prev = None
for t in range(num_steps):
    if t == 0 or prev is None:
        out = engine.step(state, t)  # full compute
    else:
        feats = compute_feats(out, prev)  # vectorized on GPU
        stable_tok = (feats.cos >= tau) | (feats.dl2 <= rho)
        counters = update_counters(stable_tok)
        tok_freeze = (counters >= m) & (feats.margin >= gamma_margin)

        layer_mask = compute_layer_mask(t, M)  # shallow layers throttled

        # IMPORTANT: sub-batch only unfrozen tokens; skip heavy matmuls for frozen
        out = engine.step(
            state, t,
            token_mask = ~tok_freeze,
            layer_mask = layer_mask
        )

        # Watchdog: if drift, recompute full for affected tokens
        alert = (feats.cos < min_cos) | (feats.kl > max_kl)
        if alert.any():
            out = engine.step(state, t)  # override: full recompute
    prev = out
```

**P3 Learned gate (TorchScript)**

```python
with torch.no_grad():
    feats = compute_feats_batch(prev, out)   # [B, F], GPU
    p_freeze, k_len = gate(feats)            # scripted model
    tok_freeze = (p_freeze >= threshold) & (feats.margin >= gamma_margin)
    freeze_horizon = torch.clamp(k_len, max=3)
    # Update freeze windows & masks as in P2; keep watchdog
```

---

## 19. Known Gaps Fixed in v2

* **`final_token_consistency = -1.0`** → now properly defined & computed; summaries include **coverage** if Teacher outputs are missing.
* **GPU util 0.0** → fixed via NVML/dmon sampling; report `gpu_util_gpu_mean`, `gpu_util_mem_mean`.
* **Masking without speed** → require **sub-batching/layer-throttling**; verify via profiler/timers.
* **Small sample sizes** → default **n_eval** increased (see §6).
* **Overhead creep** → <**3%** gate overhead budget; GPU-vectorized features.

---

## 20. Deliverables

* **Code**: `--delta-freeze`, `--adaptive-steps`, `--router` (gate selection), **dataset profiles**.
* **W&B dashboard**: all figures + Pareto.
* **Report PDF**: 6–8 pp (+ appendix), reproducible via `scripts/make_report.sh`.

---

### Appendix — Glossary

* **cosine**: cosine similarity between hidden at steps t and t−1.
* **ΔL2 ratio**: `||h_t − h_{t-1}|| / (||h_{t-1}|| + ε)`.
* **KL**: `KL(softmax z_t || softmax z_{t-1})`.
* **entropy, margin**: uncertainty & top-1 confidence gap.
* **attention centrality**: total attention mass received by a token.

```
```
