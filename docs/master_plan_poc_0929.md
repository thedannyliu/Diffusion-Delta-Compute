````markdown
# Delta-Compute Gating for Diffusion LLM — **Master Plan (v2)**

> **Goal (1-liner).** Cut end-to-end dLLM **latency & throughput** by computing **only necessary deltas** across diffusion steps—via **rule-based freezing**, **lightweight learned gating**, and **adaptive step skipping**—while **strictly controlling task quality**.

---

## 0. TL;DR
- **Hypothesis.** Consecutive diffusion steps change a subset of tokens/layers only slightly. If we **freeze+reuse** stable parts and **skip steps** when sequence-level change is low, we save FLOPs with minimal quality loss.
- **Phases.** P0 Env → **P1 Teacher traces** → **P2 Rule gate** → **P3 Learned gate** → **P4 Adaptive step scheduling** → **P5 Oracle & baselines** → **P6 Combine + ablations + report**.
- **Primary KPI.** ≥ **1.5× latency speedup** (batch=1) at ≤ **0.2σ** drop vs Teacher (or: **Wikitext-2 PPL ≤ +3%**, **LAMBADA Acc ≥ −1pp**, **GSM8K EM ≥ −1pp**). Also report a **2×** speed tier with controlled degradation and deliver **Compute-AUC** / **skip-regret** curves against true FLOPs.
- **Validation pillars.** Headroom & floor via **Oracle skip/freeze** + **simple baselines**, **scheduler-aware LTE control**, **risk-controlled gates with conformal δ ≤ 1%**, and **stress/OOD segmentation**.

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
**Speed & compute**
- **Latency** (ms/sequence, batch=1): target ≥ **1.5×** vs Teacher.
- **Throughput** (seq/s @ batch=4–8).
- **True FLOPs / MMA counts** (per-layer & total) relative to Teacher.
- **Compute-AUC** (area under Quality vs %FLOPs) and **skip-regret** (ΔFLOPs to match Teacher-quality) across operating points.

**Quality**
- **Wikitext-2 PPL**: Δ% = `(PPL_accel − PPL_teacher) / PPL_teacher × 100%` ≤ **+3%**.
- **LAMBADA-open Accuracy**: Δpp ≥ **−1pp** (absolute percentage-point).
- **Tiny GSM8K EM**: Δpp ≥ **−1pp**.

**Teacher similarity (shape preservation)**
- **Final token consistency (mean)**: fraction of positions whose final **generated token** matches Teacher. **Must** be reported.
- (Optional) **Final logit KL (mean)** at last step; aim ≤ **0.02**.
- (Optional) **Sequence exact match rate** (whole output string equality).
- **Risk-controlled false-positive rate** δ for “unsafe-to-skip but predicted safe” tokens (target δ ≤ **1.0%** with conformal calibration).

**Statistical reporting**
- For **Acc/EM**: 95% **binomial CI**.
- For **PPL**: bootstrap 95% CI over per-token NLL.
- Always report **p50/p90 latency**, **Quality–vs–FLOPs Pareto**, and include **Compute-AUC / skip-regret** tables with CIs.

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
* **Long-context scratchpads** (e.g., GSM8K-CoT 512 tokens, BookSum paragraphs)

**Tier C — Stretch**

* **GSM8K full dev**, larger MMLU slice
* **Rare-word LAMBADA** and **entity-heavy cloze** subsets
* **Cross-model eval** on second dLLM backbone (e.g., SDXL-text vs DiT-based LLM)

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
* **Noise-normalized deltas**: divide Δ features by **σₜ** (diffusion noise schedule) and log **step index / σₜ** for conditioning
* **Logit KL** KL(softmax zₜ || softmax zₜ₋₁) per token
* **Entropy**, **margin** = p(top1) − p(top2)
* **Attention centrality** (sum of incoming attention); **attention entropy**; **community IDs** from attention graph clustering
* **Per-layer timing** (timers or `torch.profiler` on a small subset)
* **Local truncation error probes**: dual-update residual (Euler vs Heun) for use in adaptive scheduling

### 7.2 Persisted artifacts

* `reports/artifacts/teacher_outputs/*.jsonl` → **final outputs** per prompt
* `reports/artifacts/features/teacher_features_*.npz` → token features (for P3) including σₜ, normalized deltas, attention communities
* `reports/runs/teacher_summary.json` → latency/throughput/p50/p90, GPU mem, CIs, true FLOPs, LTE stats
* `reports/figures/teacher_mse_layers_{timestamp}.png` & `teacher_mse_layers_line_{timestamp}.png` → **per-layer MSE heatmap + layer trend line** (new in 2025-10 update)

### 7.3 **Correct** consistency metrics (fix for `-1.0`)

* **Token-level final consistency** (per prompt):
  `consistency = (# positions with generated token == Teacher token) / length`
  Average across prompts → `final_token_consistency_mean`.
* **Sequence exact match**: whole string equality (0/1), averaged.
* (Optional) **Final logit KL** mean across positions.

> If Teacher outputs are missing, **do not** write `-1.0`; log `"missing_teacher_output": true` and skip that prompt. Include **coverage** (fraction of prompts with valid Teacher comparisons) in the summary.

### 7.4 Visualizations (richer set)

1. **Layer×Step heatmaps**: cosine (mean ± IQR), ΔL2, logit-KL, LTE residuals
2. **Layer MSE visuals**: heatmap across steps + **layer-axis line plot** of mean MSE (new)
3. **Histograms & violin plots**: cosine / ΔL2 (raw & σₜ-normalized) across all (L, t, i)
4. **Token stability lengths**: distribution of **max consecutive stable steps** (by τ/ρ & margin γ)
5. **Per-layer time breakdown**: stacked bars of attention/FFN/other + true FLOPs
6. **Centrality/community vs “freeze probability”** scatter (overlay P2/P3 decisions later)
7. **Latency tails**: p50/p90 density plots per mode (Teacher later vs P2/P3)
8. **Compute-AUC & skip-regret** curves derived from sweep over thresholds

---

## 8. P2 — **Rule-Based Gating** (Baseline, pre-configured)

**Idea.** Freeze tokens (and optionally shallow layers) that look **stable**; **watchdog** prevents drift.

### 8.1 Algorithm (per step)

1. Compute features for **unfrozen** tokens (cos, ΔL2, KL, entropy, margin) normalized by **σₜ** and conditioned on `step_frac`.
2. Update **stability counters**; if `(cos ≥ τ) OR (ΔL2 ≤ ρ)` for **m** consecutive steps, set **freeze window** of **K** steps.
3. Build `token_mask` and optional **`layer_mask`** for shallow layers (see 8.2).
4. **Watchdog** each step: if `(cos < min_cos)` OR `(KL > max_kl)` for any token in window, **rollback & recompute** immediately (see 8.6).
5. Update **conformal risk score** `s = max{ΔL2_norm / σ̂, KL, 1−cos}` and enforce `P(s ≥ τ_risk | unsafe) ≤ δ` using calibration profile (see 8.5).

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
* **Risk guarantee**: report conformal δ (target ≤1.0%) and realized false-positive count

### 8.5 Conformal risk control (δ-guardrail)

* Build monotone **risk score** `s(t, i) = max{ΔL2_norm / σ̂_t, KL, 1 − cos}` per token.
* On held-out calibration split, compute quantile `q_δ` such that `Punsafe(s ≤ q_δ) ≤ δ` with δ defaults (**Wikitext 0.5%**, **LAMBADA 0.75%**, **GSM8K 1.0%**).
* Gate decision: only freeze tokens with `s ≤ q_δ` and margin ≥ γ.
* Log **coverage**, **empirical δ**, and **interval updates** whenever model drifts (use sliding-window recalibration of size 2k prompts).

### 8.6 Rollback-safe watchdog

* Maintain **frozen-state ring buffers** for each token/layer window (depth = K).
* When watchdog triggers, restore the latest unfrozen hidden/logit state before re-running compute; prevents silent drift.
* Emit **rollback counts** and **time penalty** to confirm overhead ≤ 3%.

### 8.7 Attention-graph-aware freezing

* Cluster tokens via attention centrality/community detection (Louvain on averaged attention graph).
* Freeze clusters jointly; if any **high-centrality anchor** remains active, keep its neighborhood unfrozen.
* Provide **community-level skip ratio** tables and visualize in failure gallery heatmaps.

### 8.8 Budget-aware controller (per-sequence knapsack)

* Accept optional **budget fraction** `β` (default 0.60 of Teacher FLOPs).
* Score candidate skips by `benefit = expected_FLOPs_saved / risk_score` and greedily select until budget is met or risk exceeds δ.
* Produce **Quality@Budget** curves (β ∈ {0.40, 0.50, 0.60, 0.80}).

---

## 9. P3 — **Lightweight Learned Gate**

**Supervision.** Label (t, i) `safe_to_freeze` if freezing for next **K** steps **doesn’t change task success** or final logits **argmax/score crossing** and satisfies surrogate checks:

1. Final-token argmax unchanged when only `(t, i)` is frozen.
2. Last-step logit margin remains ≥ `γ_local = 0.07`.
3. Watchdog would not trigger within horizon K (oracle traces).

Treat ambiguous cases as **unknown** and exclude from training.

**Features.** `[cos_norm, ΔL2_norm, KL, entropy, margin, attn_sum, attn_entropy, sigma_t, step_frac, layer_band, community_id, risk_score, budget_remaining]`.

**Model.** Logistic or 2-layer MLP (hidden=64), **< 50k params**, **TorchScript**, with optional **monotone calibration head** mapping logits → risk score.

**Training.**

* Split by prompts; early stop on **AUROC** (dev) and monitor **risk-calibrated precision**.
* Handle imbalance (focal loss or `pos_weight`).
* **Calibrate** with ROC/PR + conformal δ (match §8.5); sweep thresholds to populate Compute-AUC curve.
* Fit optional **sequential SPRT** parameters (`α = 0.005`, `β = 0.005`) for moving risk score decisions.

**Deployment.**

* **Vectorized** feature computation on GPU; batch predictions.
* Output `(p_freeze, freeze_length ∈ {1,2,3}, risk_score)`; retain **watchdog** and SPRT guard.

**Calibration targets (per dataset).**

* **Wikitext-2**: precision ≥ 0.98, **skip ≈ 0.45–0.55**, δ ≤ 0.5%
* **LAMBADA**: precision ≥ 0.995, **skip ≈ 0.25–0.40**, δ ≤ 0.75%
* **GSM8K Tiny**: precision ≥ 0.995, **skip ≈ 0.15–0.30**, δ ≤ 1.0%

**Visuals.**

* **ROC/PR curves** with operating point and δ-bound
* **Calibration curves** (pred vs empirical)
* **Per-layer & community freeze maps**
* **Latency tails** vs P2/P1
* **Quality@Budget** overlays vs Rule Gate baseline

---

## 10. P4 — **Adaptive Step Scheduling** (sequence-level, LTE-controlled)

**Signals.**

* **Local truncation error (LTE)** via **Euler vs Heun** hidden/logit updates.
* **Noise-aware drift**: mean σₜ-normalized KL / entropy slope.

**Numeric controller.**

1. For each step `t`, compute candidate Euler update `ĥ_{t+k}` (stride `k`) and corrected Heun update `ḣ_{t+k}`.
2. Estimate `LTE = ||ḣ_{t+k} − ĥ_{t+k}||₂ / (||h_t||₂ + ε)` and compare with tolerance `ε_lte`.
3. If `LTE ≤ ε_lte` for **r_lte** consecutive proposals, accept stride `k+1`; otherwise reduce stride.
4. Cap total skipped steps by global **budget** `β_steps = 0.20` unless overridden by knapsack controller (§8.8).
5. Apply **backstop**: if risk score δ rises above bound, force stride=1 until recovered.

**Defaults.** `ε_lte = 1.5e-2`, `r_lte = 2`, initial stride `k = 1`, maximum stride `k_max = 4`, total skip cap `20%`.

**Outputs.** Effective steps/sequence, Δ latency/throughput, **LTE distribution**, quality deltas, consistency, and contribution to Compute-AUC.

---

## 11. P5 — **Oracle & Baseline Sandwich**

**Purpose.** Establish upper/lower bounds to contextualize gating gains.

### 11.1 Oracle skip/freeze

* Use Teacher traces to retroactively tag steps/tokens that are **safe to skip** (no task degradation, no watchdog trigger, LTE ≤ ε_lte).
* Sweep **oracle profiles**:
  * `oracle_token` — best-case token freezing with ground-truth labels.
  * `oracle_step` — best-case stride schedule with perfect foresight.
  * `oracle_combo` — combined token+step decisions under δ ≤ 0.1%.
* Report **Pareto front (Quality vs %FLOPs)** for each oracle to quantify headroom.

### 11.2 Simple baselines (floors)

* **Fixed stride after `t₀`**: choose `t₀ ∈ {4, 6, 8}`, stride `k ∈ {2,3}`.
* **Index-only policy**: stride grows with `t` (e.g., every 4 steps increase stride by 1) without features.
* **Uniform layer throttling**: recompute every `M` steps regardless of stability (M ∈ {2,3,4}).
* **Random freeze**: freeze random subset matching our skip ratio to confirm signal value.

Log their **Compute-AUC**, **skip-regret**, and δ values. Gate is validated only if it beats all simple baselines at comparable quality.

### 11.3 Oracle-to-gate gap tracking

* Track per-dataset gap: `%FLOPs_oracle − %FLOPs_gate` at fixed quality.
* Highlight regimes where gap >5% and probe failure gallery for causes (centrality, noise scale, SPRT rejects).

---

## 12. P6 — Combine & Evaluate

Compare **Teacher**, **RuleGate**, **LearnedGate**, **Adaptive**, **Combined (P2+P3+P4)** per dataset:

* **Latency** (mean/p50/p90), **Throughput**, **GPU mem**, **skip ratios**
* **Quality** (PPL/Acc/EM + CI)
* **Consistency** (token/seq), (optional) **final logit KL**
* **Pareto fronts** (Latency vs Quality)
* **Ablations**: (i) w/wo attention features; (ii) vary **K/M**; (iii) Adaptive thresholds×stride; (iv) **Stress cohorts** (context vs generation vs scratch tokens)
* **Cohort splits**: rare words, scratch segment tokens, temperature/top-k variants, cross-model comparisons

---

## 13. Instrumentation & Overhead Budget (Fixes)

**13.1 Ensure masks save FLOPs**

* Avoid “compute then overwrite.” Implement **sub-batching** for unfrozen tokens and **layer throttling** so attention/FFN matmuls are **skipped** for frozen tokens/layers.

**13.2 GPU utilization (previously 0 → **fixed**)**

* Add **NVML** (`pynvml`) sampling thread at 200–500ms to record `gpu_util_gpu_mean`, `gpu_util_mem_mean`.
* Or background `nvidia-smi dmon -s pucvmt -d 1` and parse logs.
* Use `torch.profiler` on small subsets to map per-layer cost.

**13.3 Overhead budget**

* Gating (features + model) **< 3%** runtime.
* Features computed **vectorized on GPU**, no `.item()`/Python loops.
* Reuse buffers (`engine/cache.py`) to avoid host-device thrash.

**13.4 Micro-perf validation**

* Microbench `engine.step` with synthetic batches to map speedup vs fraction frozen.
* Plot **non-linearity threshold** (expect ≥12.5% unfrozen reduction before speedup) and **memory vs compute-bound** regimes.
* Run nightly to catch regressions and to verify masks really skip FLOPs.

---

## 14. Reporting & Visualization (richer)

**Auto-generated figures per run**

1. **Layer×Step heatmaps**: cosine, ΔL2, logit-KL (mean ± IQR)
2. **Layer MSE suite**: step heatmap + layer-axis line chart (teacher freeze targeting)
3. **Cosine/ΔL2 histograms** + **violin plots** (raw & σₜ-normalized)
4. **Token stability length** distributions
5. **Per-layer time breakdown** (Teacher vs P2/P3) incl. true FLOPs
6. **Freeze maps** (fraction frozen by layer/step/community)
7. **ROC/PR** (P3) + **calibration + conformal δ** curves
8. **Latency tails** (p50/p90 density) & **skip-ratio CDFs**
9. **Pareto fronts** (Latency vs Quality) with **95% CIs**
10. **Compute-AUC & skip-regret** sweeps across thresholds/budgets
11. **Failure case gallery**: diffs in outputs + attention centrality overlays + risk heatmaps

**Tables (CSV + dashboard)**

* `{latency_mean/p50/p90, throughput, gpu_mem, gpu_util_gpu_mean, true_flops_rel, skip_ratio, unfreeze_rate, ppl/acc/em + CI, token_consistency, seq_exact_match, logit_kl, compute_auc, skip_regret, delta_bound}`

**Final PDF** (`reports/final_<date>.pdf`)

* Abstract → Method → Experiments → Ablations → Discussion → Limitations → Appendix
* Repro via `scripts/make_report.sh`.

---

## 15. Risks & Guardrails

* **Accumulated error** → watchdog + rollback buffers + LTE backstop; cap per-step **frozen proportion**.
* **Attention coupling** → community-aware freezing; keep high-centrality neighborhoods live.
* **Distribution shift / OOD** → stress suites (long-context, scratchpads, rare words) + conformal recalibration.
* **Risk bound drift** → monitor δ in production; trigger recalibration if δ > target for 3 consecutive windows.
* **Overhead creep** → continuous profiling; block PRs exceeding **3%** gate overhead or failing microbench thresholds.
* **Budget misuse** → enforce Quality@Budget watchdog (never exceed FLOPs budget without warning).

---

## 16. Milestones & Owners

* **P0 (Day 0–0.5)**: Env + repo skeleton; local engine compiles. *(Danny)*
* **P1 (Day 0.5–1)**: Teacher traces + figures. *(Danny)*
* **P2 (Day 1–2)**: Rule gate + **dataset presets** + curves. *(Danny)*
* **P3 (Day 2–3)**: Learned gate + calibration. *(Danny)*
* **P4 (Day 3–4)**: Adaptive steps. *(Danny)*
* **P5 (Day 4–5)**: Combine + ablations + draft report. *(Danny; review: Celine/Zhenbang/Jai)*

---

## 17. Checklists (updated)

**Before each run**

* [ ] Git commit hash saved
* [ ] Configs frozen (`.yaml` copied into run dir)
* [ ] Seed fixed; deterministic dataset slice
* [ ] **Teacher outputs present** (or logged missing with coverage)
* [ ] W&B run name `{mode}-{model}-{date}-{gitsha}`
* [ ] Conformal calibration profile (`q_δ`) pinned for dataset/temperature/top-k
* [ ] FLOPs budget `β` + stride caps recorded

**After each run**

* [ ] **Quality metrics with CIs** written
* [ ] **Consistency** metrics computed (coverage reported)
* [ ] Pareto plots updated; artifacts uploaded
* [ ] **Top 3 failure cases** annotated
* [ ] Compute-AUC / skip-regret updated; δ tracked vs target
* [ ] Microbench + LTE residual sanity checks stored

---

## 18. Example Local Scripts

**`scripts/run_teacher_local.sh`**

```bash
python -m src.run \
  --mode teacher \
  --tasks_cfg configs/tasks.yaml \
  --paths_cfg configs/paths.yaml \
  --out_dir reports \
  --oracle_eval true \
  --dump_features_dir reports/artifacts/features \
  --lte_probe heun
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
  --risk_delta 0.006 \
  --budget_fraction 0.60 \
  --out_dir reports
```

**`scripts/run_learned_gate_local.sh`**

```bash
python -m src.run \
  --mode learned_gate \
  --tasks_cfg configs/tasks.yaml \
  --paths_cfg configs/paths.yaml \
  --gate_weights reports/artifacts/gates/learned_gate_latest.ts \
  --calibration_target precision=0.98,skip=0.5,delta=0.005 \
  --sprt_alpha 0.005 --sprt_beta 0.005 \
  --dataset wikitext2 \
  --out_dir reports
```

**`scripts/run_adaptive_steps_local.sh`**

```bash
python -m src.run \
  --mode adaptive \
  --tasks_cfg configs/tasks.yaml \
  --paths_cfg configs/paths.yaml \
  --lte_eps 0.015 --lte_min_consec 2 --max_stride 4 \
  --adaptive_budget 0.20 \
  --out_dir reports
```

---

## 19. Pseudocode (fixed & clarified)

**P2 Rule-based (token+layer)**

```python
state = engine.encode(prompt)
rollback = RollbackBuffer(depth=K)
prev = None
for t in range(num_steps):
    sigma_t = noise_schedule[t]
    if t == 0 or prev is None:
        out = engine.step(state, t)
    else:
        feats = compute_feats(out, prev, sigma_t)  # GPU, normalized by σ_t
        risk = compute_risk_score(feats)
        stable_tok = ((feats.cos >= tau) | (feats.dl2 <= rho)) & (feats.margin >= gamma_margin)
        tok_freeze = stable_tok & (risk <= risk_quantile)

        layer_mask = compute_layer_mask(t, M)
        budget_mask = budget_controller.select(tok_freeze, risk)

        rollback.save(state, t, tok_freeze)

        out = engine.step(
            state,
            t,
            token_mask=~budget_mask,
            layer_mask=layer_mask,
            stride=stride_schedule.current_stride,
        )

        alert = (feats.cos < min_cos) | (feats.kl > max_kl) | (risk > risk_quantile)
        if alert.any():
            state = rollback.restore()
            out = engine.step(state, t)
            stride_schedule.reset()
    stride_schedule.update(feats, risk)
    prev = out
```

**P3 Learned gate (TorchScript)**

```python
with torch.no_grad():
    feats = compute_feats_batch(prev, out, sigma_t)  # [B, F]
    p_freeze, k_len, risk = gate(feats)
    sprt_accept = sprt.test(risk_series=state.sprt_state)
    tok_freeze = (p_freeze >= threshold) & (risk <= risk_quantile) & sprt_accept
    freeze_horizon = torch.clamp(k_len, max=3)
    budget_mask = budget_controller.select(tok_freeze, risk)
    apply_freeze_masks(budget_mask, freeze_horizon)
    watchdog.register(feats, risk)
```

---

## 20. Known Gaps Fixed in v2

* **`final_token_consistency = -1.0`** → now properly defined & computed; summaries include **coverage** if Teacher outputs are missing.
* **GPU util 0.0** → fixed via NVML/dmon sampling; report `gpu_util_gpu_mean`, `gpu_util_mem_mean`.
* **Masking without speed** → require **sub-batching/layer-throttling**; verify via profiler/timers + microbench suite (§13).
* **Small sample sizes** → default **n_eval** increased (see §6).
* **Overhead creep** → <**3%** gate overhead budget; GPU-vectorized features.
* **No headroom/floor** → Oracle + baseline sandwich (P5) codified with Compute-AUC & skip-regret reporting.
* **Heuristic stride changes** → LTE-controlled adaptive schedule with rollback guarantees (P4).
* **Unbounded risk** → Conformal δ-bounded gating with SPRT fallback (P2/P3).
* **Noise-scale drift** → σₜ normalization baked into features, labels, and calibration.

---

## 21. Deliverables

* **Code**: `--delta-freeze`, `--adaptive-steps`, `--router`, `--oracle-eval`, `--baseline-eval`, conformal calibration scripts, microbench tooling, budget controller.
* **Metrics**: Compute-AUC, skip-regret tables, δ guarantees, true FLOPs traces, LTE histograms.
* **Artifacts**: Oracle/baseline Pareto plots, Quality@Budget curves, OOD cohort breakdowns, microbench sweeps, budget controller logs.
* **Cross-model evidence**: replicate on ≥2 dLLM architectures and decoding settings (temperature 0.7 vs 1.0, top-k ∈ {20,50}).
* **W&B dashboard**: all figures + Pareto + Quality@Budget + δ tracking.
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
