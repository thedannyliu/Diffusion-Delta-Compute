# Delta-Compute Gating for Diffusion LLM — **Master Plan (v3)**

> **One-line goal.** Reduce end-to-end dLLM **latency & FLOPs** by computing **only the necessary deltas** across diffusion steps—via **rule-based freezing (P2)**, a **lightweight learned gate (P3)**, and **adaptive step skipping (P4)**—while **formally controlling risk** and **preserving task quality**.

---

## 0) TL;DR (what changed vs v2)

* **Corrected metrics & math:** formal definitions for **δ_frozen**, **coverage**, **true FLOPs**, **Compute-AUC**, and **skip-regret**; unified latency/GPU-util measurement protocol.
* **Real speed, not pretend speed:** engines **must skip matmuls** (sub-batch/block-sparse). Added a **micro-perf suite** and **per-layer cost model**.
* **Step-wise thresholds:** P2/P3 use **per-step quantiles** of `(1−cos)`, `ΔL2_norm`, and `KL` logged in P1; conformal risk is **per-step** with smoothing/clip.
* **Risk you can trust:** all risk is reported as **δ_frozen** (only over actually skipped/frozen positions); added **KL hard guard** and **rollback budget ≤3%**.
* **Adaptive that doesn’t over-skip:** P4 uses **normalized LTE** + **risk dual condition**, **step-wise max stride**, **tail-guard**, and **β_steps hard cap**.
* **Oracle/Baseline sandwich:** mandatory **Compute-AUC** and **Oracle-to-Gate gap@Quality** reporting; gate must **beat fixed-stride/index-only/layer-throttle**.

---

## 1) Scope & Non-Goals

**In scope.** Inference-time only (no backbone retrain). Per-token/per-layer freezing and per-sequence step skipping for locally loaded diffusion LLMs.

**Not in v3.** New backbone training; heavy distillation; production hardening beyond a reproducible research prototype.

---

## 2) Success Criteria & **Formal Metric Definitions**

### 2.1 Speed & Compute KPIs

* **Latency** (ms/seq, batch=1): target **≥1.5×** speedup vs Teacher; report **mean/p50/p90**.
* **Throughput** (seq/s @ batch 4–8).
* **True FLOPs / MMA counts.** Prefer profiler counts. If estimating, track per-layer costs (see Appendix A).
* **Compute-AUC.** Area under the curve of **Quality vs %FLOPs** (FLOPs normalized to Teacher=100%). Use linear interpolation on the **FLOPs axis**; report 95% bootstrap CI.
* **Skip-regret.** At a fixed quality (\mathcal{Q}^*),
  [
  \text{regret}=\min_{j:, \text{Quality}_j\ge \mathcal{Q}^*}\ %{\rm FLOPs}_j - 100% .
  ]

### 2.2 Quality KPIs (hold these at Teacher or within budgeted deltas)

* **Wikitext-2 PPL**: Δ% ≤ **+3%**.
* **LAMBADA-open Acc**: ≥ **−1 pp**.
* **Tiny GSM8K EM**: ≥ **−1 pp**.
  All with **95% CI** (binomial for Acc/EM; bootstrap for PPL).

### 2.3 Shape-preservation & Risk

* **Final token consistency (mean)**: per-prompt token match rate vs Teacher, then average across prompts; also report **coverage** (fraction of prompts with valid Teacher comparison).
* **Sequence exact-match** (optional) & **final logit KL** (goal ≤ 0.02).
* **δ_frozen (core risk)**: false-positive rate **only over actually skipped/frozen** items
  [
  \delta_{\text{frozen}}=\frac{#{(t,i)\in\mathcal{F}:\ \text{unsafe}(t,i)\land \text{decide_skip}(t,i)}}{|\mathcal{F}|},\ \ \mathcal{F}={\text{positions actually frozen/skipped}}.
  ]
  Target: **≤1%** (dataset-dependent; see §8.5, §9).

### 2.4 Latency and GPU-util protocol

* Warm once; measure with fixed **temperature/top-k** and **batch**; collect **p50/p90**.
* NVML/dmon sampling every 200–500 ms → **gpu_util_gpu_mean**, **gpu_util_mem_mean**.
* Exclude heavy logging/visualization during timing runs.

---

## 3) Environment & Determinism

```bash
conda create -y -n dcllm python=3.10
conda activate dcllm
pip install -r requirements.txt  # match torch/cu (PyTorch ≥ 2.2)
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

```
dcllm/
  README.md
  master_plan.md

  configs/
    tasks.yaml
    thresholds.yaml         # P2 rule-gate profiles + risk settings
    adaptive.yaml           # P4 step scheduler settings
    paths.yaml

  scripts/
    run_teacher_local.sh    # P1
    run_rule_gate_local.sh  # P2
    run_learned_gate_local.sh# P3
    run_adaptive_steps_local.sh # P4
    make_report.sh

  src/
    run.py
    engine/
      local_engine.py       # step(); must support sub-batch/block-sparse masks
      cache.py              # pinned buffers, ring buffers for rollback
    probes/
      metrics.py            # cosine, ΔL2, KL, entropy, margin
      sampler.py            # LTE probe (Euler-Heun), moving stats
    gating/
      rule_gate.py          # P2
      learned_gate.py       # P3 (TorchScript)
      scheduler.py          # P4 (stride control)
      calibrate.py          # conformal q_δ(t), ROC/PR, calibration
    eval/
      tasks.py
      scorer.py
      consistency.py
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

---

## 5) Data & Evaluation

* **Tier-A smoke:** Wikitext-2 (~2k), LAMBADA-open (2k), Tiny GSM8K (1–2k).
* **Tier-B day-scale:** CNN/DM-mini, MMLU-mini (5×100), long-context scratchpads.
* **Tier-C stretch:** GSM8K full, larger MMLU, rare-word LAMBADA, cross-backbone.

Deterministic subsampling via `configs/tasks.yaml`.

---

## 6) Config Examples

### 6.1 `configs/tasks.yaml`

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

### 6.2 `configs/paths.yaml`

```yaml
data_root:  /path/to/data
models_root:/path/to/models
outputs_root:/path/to/outputs
```

### 6.3 `configs/thresholds.yaml` (P2, step-wise + risk)

```yaml
rule_gate:
  risk:
    delta_target: {wikitext2: 0.005, lambada_open: 0.0075, gsm8k_tiny: 0.01}
    kl_guard: 0.003
    smoothing: ema_0p9
    clip_quantiles: [0.10, 0.99]  # protect q_δ(t) from collapse

  stepwise_thresholds:
    wikitext2:
      one_minus_cos_q: 0.88      # use q_0.88^(t) of (1-cos) per step
      dl2_norm_q:     0.40       # use q_0.40^(t) of ΔL2_norm per step
      margin_gamma:   0.09

  schedule:
    m_k_m:
      early: {m: 2, K: 1, M: 1}
      mid:   {m: 1, K: 2, M: 2}
      late:  {m: 1, K: 3, M: 2}
    heavy_layers: [0,1,2]   # from profiler; update per-backbone
```

### 6.4 `configs/adaptive.yaml` (P4, normalized LTE + dual condition)

```yaml
adaptive:
  use_normalized_lte: true
  lte_eps_step: [0.020,0.020,0.018,0.018,0.016,0.015,0.015,0.014,0.013,0.012,0.012,0.012]
  min_consecutive:  [4,4,3,3,2,2,2,2,2,2,2,2]
  max_stride_step:  [1,2,2,3,3,4,4,4,4,2,2,2]  # includes tail-guard
  beta_steps: 0.20
  risk_link:
    use_conformal: true
    kl_guard: 0.003
```

---

## 7) P1 — Teacher Traces & Visualizations

**Purpose.** Full-compute run; log per-step, per-layer, per-token features to expose stability & cost hotspots.

**Log** per step (t), layer (L), token (i):

* cosine((h_t,h_{t-1})), (\Delta L2_{\rm norm}=|h_t-h_{t-1}|/(|h_{t-1}|+\varepsilon))
* **σ-normalized** deltas; keep `sigma_t` and `step_frac`
* **KL** ( {\rm KL}(\text{softmax}\ z_t || \text{softmax}\ z_{t-1}) )
* entropy, margin (p_{\rm top1}-p_{\rm top2})
* attention centrality/entropy; community IDs
* **per-layer timing** (profiler on a small slice)
* **LTE probe** (Euler vs Heun residual)
* **NEW:** store **per-step quantiles** (q_p^{(t)}) for `(1−cos)`, `ΔL2_norm`, `KL`, and **phase feature** `step_mod` (e.g., modulo 4)

**Persist**

* teacher outputs (for consistency), features NPZ, quantile tables, profiler summaries.

**Visuals** (auto-save): layer×step heatmaps; histograms/violins; layer-time breakdown; token stability spans; latency tails.

---

## 8) P2 — **Rule-Based Gating**

### 8.1 Implementation requirements (speed is real)

* **Must skip matmuls**: attention/FFN compute **only** for unfrozen tokens (`sub-batch`), or via **block-sparse** routing.
* **No compute-then-overwrite.**
* Provide a micro-perf curve: `unfrozen_ratio → step() latency`; mark the non-linearity threshold (expect ≥12–15% reduction to see speedup).

### 8.2 Step-wise thresholds (from P1 quantiles)

* Freeze candidates must satisfy, for step (t):

  * ((1-\cos) \le q^{(t)}_{0.88\sim0.90})
  * (\Delta L2_{\rm norm} \le q^{(t)}_{0.30\sim0.50})
  * `margin ≥ γ` (γ=0.08–0.10)

### 8.3 Conformal risk guard (per-step)

* Risk score ( s(t,i)=\max{\Delta L2_{\rm norm}, 1-\cos, \text{KL}}).
* On calibration split, estimate **per-step** (q_\delta^{(t)}) so that (P_{\rm unsafe}(s \le q_\delta^{(t)}) \le \delta).
* Decision: `freeze ⇐ s ≤ q_δ^(t) ∧ margin ≥ γ`.
* **Smooth** (q_\delta^{(t)}) (EMA 0.9) and **clip** to ([p_{10},p_{99}]).

### 8.4 Watchdog + rollback

* If `KL > 0.003` or `(1−cos)` spikes: **rollback and recompute**.
* Keep ring buffers (depth=K). Report `rollback_count` and `rollback_time_ratio ≤ 3%`.

### 8.5 m/K/M & heavy layers

* Early: `m=2, K=1`; Mid: `m=1, K=2`; Late: `m=1, K=2–3`.
* Apply `M=2–3` to **only the costliest 2–3 layers** (from profiler); others `M=1–2`.
* Inject small **K ± 1 jitter** to avoid resonance with `M`.

### 8.6 Budget controller

* Optional FLOPs budget (\beta) (default 0.60). Greedy knapsack on **benefit/risk**.

### 8.7 Outputs & checks

* Skip ratios (token×layer×step), latency p50/p90, throughput, **true FLOPs rel**, **δ_frozen + coverage**, consistency, rollback stats.

---

## 9) P3 — **Lightweight Learned Gate**

### 9.1 Supervision (strict)

Positive if freezing (t,i) for K steps:

1. **Final argmax unchanged**,
2. **final-step margin ≥ 0.07**,
3. **watchdog would not trigger** within horizon K.
   Ambiguous → drop.

### 9.2 Features

([{\rm cos_norm}, \Delta L2_{\rm norm}, {\rm KL}, {\rm entropy}, {\rm margin}, {\rm attn_sum}, {\rm attn_entropy}, \sigma_t, {\rm step_frac}, {\rm layer_band}, {\rm community_id}, {\rm risk_score}, {\rm budget_remaining}, {\rm step_mod}])

### 9.3 Model & training

* Logistic / 2-layer MLP (hidden=64), **<50k params**, TorchScript.
* **Cost-weighted positives**: weight ∝ **FLOPs_saved**(t,i).
* Handle imbalance (focal or `pos_weight`).
* Calibrate (ROC/PR; temperature scaling or isotonic).
* Conformal δ as in §8.3; optional sequential SPRT (α=β=0.005).

### 9.4 Deployment

* Vectorized feature compute on GPU; batched inference.
* Outputs: ((p_{\rm freeze}, K_{\rm len}\in{1,2,3}, \text{risk})). Keep watchdog.

### 9.5 Operating targets (per dataset)

* Wikitext-2: precision ≥ 0.98, **skip 10–15%** (practical target; earlier “0.45–0.55” was headroom), **δ_frozen ≤ 0.5–1.0%**.
* LAMBADA: precision ≥ 0.995, skip 8–12%, δ ≤ 0.75%.
* Tiny GSM8K: precision ≥ 0.995, skip 5–10%, δ ≤ 1.0%.

### 9.6 Budget shaping

* **Mid-span budget**: t∈[4,7] get β_mid=0.15–0.20 to focus on costly mid steps.

---

## 10) P4 — **Adaptive Step Scheduling** (sequence-level)

### 10.1 Signals

* **Normalized LTE** (Euler vs Heun):
  [
  \text{LTE}*{\rm norm}=\frac{| \hat{h}^{\rm Heun}*{t+k}-\hat{h}^{\rm Euler}_{t+k}|_2}{|h_t|_2+\varepsilon}
  ]
  or scaled by (\sigma_t).
* Drift proxies: σ-normalized KL, entropy slope.

### 10.2 Controller (dual condition)

* If **for r consecutive proposals**: `LTE_norm ≤ ε(t)` **and** `risk ≤ q_δ(t)`, then **increase stride** by 1 (up to **step-wise max**).
* If `KL > 0.003` or `risk > q_δ(t)`, set `stride=1` and reset counter.

### 10.3 Step-wise caps & tail-guard

* `max_stride_step`: t≤3→2, 4–6→3, ≥7→4.
* **Tail-guard**: last 1–2 steps force `stride ≤ 2`.

### 10.4 Global step budget

* **β_steps = 0.20** (hard cap per sequence).

### 10.5 Co-scheduling with P3

* Allow `stride+1` only if **P3 coverage on current step ≥ threshold** (e.g., ≥10% of tokens frozen safely).

---

## 11) P5 — Oracle & Baseline Sandwich

### 11.1 Oracle variants

* `oracle_token` (best possible token freezing),
* `oracle_step` (best possible stride schedule),
* `oracle_combo` (joint; **δ ≤ 0.1%**).
  Report **Pareto fronts** and **headroom**.

### 11.2 Baselines (floors)

* Fixed stride after (t_0) with (k\in{2,3});
* Index-only stride growth;
* Uniform layer throttling (M\in{2,3,4});
* Random freeze (match skip ratio).

### 11.3 Mandatory reporting

* **Compute-AUC** (95% CI);
* **Oracle-to-Gate gap@Quality**: at fixed quality (Teacher), `%FLOPs_oracle − %FLOPs_gate` **≤ 5–8%**;
* Show gate **beats all baselines at comparable quality**.

---

## 12) P6 — Combine & Evaluate

Compare **Teacher**, **RuleGate**, **LearnedGate**, **Adaptive**, and **Combined (P2+P3+P4)**:

* Latency (mean/p50/p90), throughput, GPU mem/util, **true FLOPs**, skip ratios (token/step), **δ_frozen**, rollback stats.
* Quality (PPL/Acc/EM + 95% CI), consistency, final logit KL (opt).
* Pareto (Latency vs Quality).
* Ablations: w/wo attention features; vary K/M; LTE thresholds×stride; stress cohorts (rare words, scratch tokens, long-context); decoding variants; cross-backbone.

---

## 13) Instrumentation & Overhead Budget

* **Masks must save compute.** Sub-batch/block-sparse only; never compute-then-overwrite.
* **NVML/dmon** sampling enabled; store `gpu_util_gpu_mean`, `gpu_util_mem_mean`.
* **Overhead budget**: features + gating **<3%** runtime; vectorize, avoid `.item()`; reuse GPU buffers.
* **Micro-perf validation:** synthetic microbench of `step()` vs unfrozen ratio; plot non-linearity threshold and compute- vs memory-bound regimes. Nightly regression checks.

---

## 14) Reporting & Visualization (minimum set per run)

* **δ_frozen–Coverage curves** (overall & per-step) → choose operating points.
* **Skip (token/step) vs latency p50/p90** scatter (expose non-linearity threshold).
* **Stride trace** overlaid with **LTE_norm & risk** (verify triggers).
* **Per-layer time share** (Teacher vs Gate) to prove costly layers throttled.
* Layer×step heatmaps: cosine, ΔL2, KL; histograms/violins (raw & σ-norm).
* Token stability spans; latency tails; freeze maps; ROC/PR & calibration (P3).
* **Compute-AUC** & **skip-regret** with 95% CI.
* **Failure gallery** with attention overlays and risk heatmaps.

---

## 15) Risks & Guardrails

* **Accumulated error** → watchdog+rollback; cap per-step frozen proportion.
* **Attention coupling** → community-aware freezing; keep high-centrality anchors live.
* **Distribution shift/OOD** → stress suites + sliding-window recalibration of (q_\delta^{(t)}).
* **Risk drift** → alert if **δ_frozen** exceeds target for 3 consecutive windows.
* **Overhead creep** → block PRs that exceed **3%** gate overhead or fail micro-perf thresholds.
* **Budget misuse** → Quality@Budget watchdog; never exceed FLOPs budget silently.

---

## 16) Milestones

* **P0 (Day 0–0.5)** Env & skeleton; engine compiles.
* **P1 (0.5–1)** Teacher traces + quantiles + figures.
* **P2 (1–2)** Rule gate + step-wise thresholds + δ_frozen reporting.
* **P3 (2–3)** Learned gate + cost-weighted training + calibration.
* **P4 (3–4)** Adaptive (normalized LTE, dual condition, tail-guard, β_steps).
* **P5 (4–5)** Oracle/Baseline, Compute-AUC, gap@Quality; combine + ablations + draft report.

---

## 17) Checklists

**Before each run**

* [ ] Git SHA pinned; configs copied into run dir.
* [ ] Seed fixed; deterministic dataset slice.
* [ ] Teacher outputs present (or `missing_teacher_output=true`), **coverage** logged.
* [ ] NVML/dmon enabled; warmup excluded from timing.
* [ ] Conformal profiles (q_\delta^{(t)}) pinned for dataset/temperature/top-k.
* [ ] FLOPs budget `β` + stride caps recorded.

**After each run**

* [ ] Quality (with CI) & **δ_frozen + coverage** written.
* [ ] Consistency & (opt) final logit KL.
* [ ] True FLOPs rel, p50/p90 latency, throughput, GPU util.
* [ ] Rollback counts & **rollback_time_ratio**.
* [ ] Pareto & Compute-AUC updated; **Oracle-gap@Quality** computed.
* [ ] Micro-perf & LTE residual sanity checks stored.
* [ ] Top-3 failure cases annotated.

---

## 18) Example Scripts

```bash
# P1
python -m src.run \
  --mode teacher \
  --tasks_cfg configs/tasks.yaml \
  --paths_cfg configs/paths.yaml \
  --out_dir reports \
  --oracle_eval true \
  --dump_features_dir reports/artifacts/features \
  --lte_probe heun
```

```bash
# P2
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

```bash
# P3
python -m src.run \
  --mode learned_gate \
  --tasks_cfg configs/tasks.yaml \
  --paths_cfg configs/paths.yaml \
  --gate_weights reports/artifacts/gates/learned_gate_latest.ts \
  --calibration_target precision=0.98,delta=0.005 \
  --sprt_alpha 0.005 --sprt_beta 0.005 \
  --dataset wikitext2 \
  --out_dir reports
```

```bash
# P4
python -m src.run \
  --mode adaptive \
  --tasks_cfg configs/tasks.yaml \
  --paths_cfg configs/paths.yaml \
  --adaptive_cfg configs/adaptive.yaml \
  --out_dir reports
```

---

## Appendix A — Cost model (when profiler is unavailable)

For a transformer-style block with (L) layers, sequence length (S), heads (H), key dim (d_k), model dim (d), FFN hidden (d_{\rm ff}):

[
\begin{aligned}
\text{FLOPs}*{\rm attn} &\approx 2L\cdot H\cdot d_k\cdot S^2 \quad (\text{QK}^\top + {\rm softmax}\cdot V) \
\text{FLOPs}*{\rm ffn}  &\approx 2L\cdot S\cdot d\cdot d_{\rm ff}
\end{aligned}
]
Use the **measured per-layer shares** to weight P2/P3 decisions and to compute expected **FLOPs_saved** for cost-weighted training.

---

## Appendix B — Pseudocode (final)

**P2 Rule gate**

```python
state = engine.encode(prompt)
rollback = RollbackBuffer(depth=K)
prev = None
for t in range(num_steps):
    out = engine.step(state, t) if t == 0 or prev is None else out
    if t > 0:
        feats = compute_feats(prev, out, sigma_t)  # GPU, σ-normalized
        s = risk_score(feats)                      # max{dl2_norm, 1-cos, kl}
        q_delta_t = conformal_quantile(t)          # smoothed & clipped
        stable_tok = ((1-feats.cos) <= q_one_minus_cos(t)) \
                   & (feats.dl2_norm <= q_dl2(t)) \
                   & (feats.margin >= gamma)
        tok_freeze = stable_tok & (s <= q_delta_t)

        layer_mask = layer_throttle_mask(t, M, heavy_layers)
        budget_mask = budget_controller.select(tok_freeze, s)

        rollback.save(state, t, tok_freeze)
        out = engine.step(
            state, t,
            token_mask=~budget_mask,
            layer_mask=layer_mask,
            stride=stride_schedule.current_stride,
            # Engine must sub-batch/block-sparse here
        )

        alert = (feats.kl > kl_guard) | ((1-feats.cos) > spike_guard)
        if alert.any():
            state = rollback.restore()
            out = engine.step(state, t)
            stride_schedule.reset()
    stride_schedule.update(feats, s)
    prev = out
```

**P3 Learned gate**

```python
with torch.no_grad():
    feats = compute_feats_batch(prev, out, sigma_t)  # [B, F]
    p_freeze, k_len, s = gate(feats)                 # TorchScript
    sprt_ok = sprt.test(update=s)                    # optional
    accept = (p_freeze >= theta_step[t]) & (s <= q_delta(t)) & sprt_ok
    horizon = torch.clamp(k_len, max=3)
    budget_mask = budget_controller.select(accept, s)
    apply_freeze_masks(budget_mask, horizon)
    watchdog.register(feats, s)
```

**P4 Adaptive**

```python
for t in range(num_steps):
    lte = lte_norm(euler, heun, state)              # normalized
    risk_ok = (risk <= q_delta(t)) and (kl <= kl_guard)
    if lte <= eps_step[t] and risk_ok and consec >= min_consec[t] \
       and stride < max_stride_step[t] and steps_skipped < beta_steps_cap:
        stride += 1
        consec = 0
    elif not risk_ok:
        stride = 1; consec = 0
    else:
        consec += 1
    if is_tail(t):
        stride = min(stride, 2)                      # tail-guard
```

---

**That’s the complete v3 markdown.** It bakes in the corrected metrics, gating math, adaptive logic, YAMLs, and reporting needed to hit the **≥1.5×** latency target with **δ_frozen ≤ 1%** at stable quality.
