# Delta-Compute Gating for Diffusion LLM — **master_plan.md**

> **Goal (1‑liner)**: Improve end‑to‑end dLLM inference **latency & throughput** by computing **only the necessary deltas** across diffusion steps — via **rule‑based freezing**, **learned gating**, and **adaptive step skipping** — while maintaining task quality.

---

## 0. TL;DR for busy readers
- **Hypothesis**: Many tokens/layers change little across consecutive diffusion steps. If we **freeze/copy** stable parts and **skip steps** when global change is small, we can save FLOPs with negligible quality loss.
- **Phases**: (P0) Environment & baselines → (P1) Teacher traces & visualizations → (P2) Rule‑based gating → (P3) Lightweight learned gate → (P4) Adaptive step scheduling → (P5) Combine + ablations + report.
- **KPI**: ≥**1.5×** latency speedup **with ≤0.2σ quality drop** vs teacher on chosen tasks; add results for **2×** speed with controlled degradation.

---

## 1. Scope & Non‑Goals
**In‑scope**
- Inference‑time only (no re‑training of the backbone dLLM).  
- Per‑token & per‑layer **delta‑compute** (freeze/copy) and **step skipping**.  
- General API that wraps any dLLM engine exposing **per‑step hidden states/logits/attn**.

**Out‑of‑scope** (v1)
- Training new diffusion LLMs from scratch.  
- Heavy distillation or fine‑tuning of the backbone.  
- Full productionization; we deliver a research‑grade prototype with clean flags & scripts.

---

## 2. Success Criteria & KPIs
- **Latency**: wall‑clock per sequence reduced by **≥1.5×** at batch=1 on A100/H100.  
- **Throughput**: tokens/s (or sequences/s) improved similarly at batch=4–8.  
- **Quality**: For each task, metric drop ≤ **0.2 standard deviations** from teacher (or absolute: Perplexity +≤3%, ROUGE/LAMBADA/GSM8K EM −≤1%).  
- **Ablations**: clear FLOPs/latency–quality curves; sensitivity over thresholds τ (cos), ρ (ΔL2), γ (margin), K (freeze horizon), M (recompute cadence).

---

## 3. Environment Setup
> Target: **CUDA 12.x**, **PyTorch ≥2.2**. Assumes GT ICE/PACE clusters with Slurm.

```bash
# Create env
conda create -y -n dcllm python=3.10
conda activate dcllm

# Install
pip install -r requirements.txt
```

**Cluster prerequisites**
- Slurm modules (example): `module load cuda/12.1 anaconda/2023.03`.
- Set W&B: `export WANDB_ENTITY=<team>`; `export WANDB_PROJECT=dcllm-delta`.
- Determinism: `torch.backends.cudnn.benchmark=False`; fixed `seed=42`.

---

## 4. Repository Layout (proposed)
```
dcllm/
  README.md
  master_plan.md                # this file
  configs/
    tasks.yaml                  # eval suites & dataset slices
    thresholds.yaml             # τ, ρ, γ, K, M, etc.
  scripts/
    setup_env.sh
    run_teacher.sh              # full compute baseline
    run_rule_gate.sh            # P2
    run_learned_gate.sh         # P3
    run_adaptive_steps.sh       # P4
    sbatch_teacher.sbatch       # Slurm examples
  src/
    engine/
      base_engine.py            # unified interface for any dLLM
      d2f_engine.py             # wrapper for D2F/dLLM impl
    probes/
      hooks.py                  # capture per-step/layer activations
      metrics.py                # cosine, ΔL2, KL, entropy, margin
    gating/
      rule_gate.py              # P2
      learned_gate.py           # P3
      scheduler.py              # P4 adaptive step skipping
    viz/
      plots.py                  # heatmaps, histograms, curves
    eval/
      tasks.py                  # loaders & metrics
      scorer.py                 # PPL/ROUGE/EM/acc
    utils/
      logging.py seed.py timer.py profile.py
  notebooks/                    # quick EDA/viz
  reports/
    figures/                    # auto-saved
    runs/                       # tables (csv)
```

---

## 5. Data & Evaluation Tasks
We keep **three tiers** for quick iteration:

1) **Smoke** (≤1h total):  
- **Wikitext‑2** (perplexity proxy); **LAMBADA‑open** (acc); **Tiny GSM8K** 1k subset (EM).  

2) **Day‑scale**:  
- **CNN/DM‑mini** for summarization (ROUGE‑L); **MMLU‑mini** 5 subjects×100 qs (acc).  

3) **Stretch**: full GSM8K dev; larger MMLU slice.

> All datasets read via `datasets` hub; subsampling controlled in `configs/tasks.yaml`. For PoC we prioritize: **Wikitext‑2 PPL**, **LAMBADA‑open acc**, **Tiny GSM8K EM**.

---

## 6. Unified dLLM Engine API
We avoid coupling to a specific repo. Any diffusion‑style text model should implement:

```python
class DiffusionEngine(BaseEngine):
    def encode_prompt(self, prompt: str) -> EngineState: ...
    def step(self, state: EngineState, t: int, compute_mask=None) -> StepOutputs:
        """
        Runs one diffusion step, returning:
        - hidden_by_layer: List[Tensor]  # [n_layers, seq, hidden]
        - logits: Tensor                 # [seq, vocab]
        - attn_stats (optional): Dict    # e.g., per-token attention sums/entropy
        - aux: Dict                      # scheduler info, noise level, etc.
        """
    def decode(self, state: EngineState) -> str: ...
    def num_steps(self) -> int: ...
```

- `compute_mask` (Seq‑len bool) allows **per‑token freeze**: masked tokens reuse previous step’s hidden/logits.
- For **per‑layer reuse**, we pass an optional `layer_mask` via `aux`.

---

## 7. P1 — Teacher Traces & Visualizations (Day 0–1)
**What**: Run *full compute* (no skipping) to log per step & per layer tensors.

**Logged per step** (for each layer L and token i):
- **Cosine** between `h_{t}^{L,i}` and `h_{t-1}^{L,i}`.  
- **ΔL2 ratio**: `||h_t - h_{t-1}||_2 / ||h_{t-1}||_2`.  
- **Logit KL**: `KL(softmax z_t || softmax z_{t-1})` per token.  
- **Entropy** of logits; **margin** = p(top1) − p(top2).  
- **Attention centrality**: sum over heads of attention mass received by token i; and attn entropy.

**Figures** (auto‑saved to `reports/figures` & W&B):
- **Heatmaps** (layer × step) for cosine / ΔL2 (mean ± 25–75% IQR).  
- **Histograms** of cosine across all (layers, steps, tokens) with mean line.  
- **Token‑wise stability**: distribution of **max consecutive stable steps** where `(argmax stable) ∧ (margin>γ)`.

**Deliverable**: `reports/runs/<date>_teacher.jsonl` (aggregates) + figure set.

**Performance logging**: record wall‑clock per‑sequence latency (batch=1), throughput (seq/s at batch=4–8), and GPU utilization snapshot per run.

---

## 8. P2 — Rule‑based Gating (Baseline)
**Heuristics**
- Freeze **if** `(cos ≥ τ) OR (ΔL2 ≤ ρ)` holds for **m consecutive steps**. Start with `τ=0.95, ρ=0.10, m=2`.
- **Freeze horizon**: `K=2` steps (extend to 3–4 in ablation).  
- **Watchdog** (health‑check): every K steps recompute; **unfreeze immediately** if `(cos<0.90) OR (KL>0.02)`.
- **Layer reuse**: For shallow layers (e.g., L1–L3) where teacher traces show high cosine, only recompute every `M` steps (M=2 or 3); deeper layers always compute.

**Algorithm (per step)**
1. Compute features for tokens not frozen.  
2. Update **stability counters**; start/extend freeze windows.  
3. Build `compute_mask` & optional `layer_mask`; call `engine.step`.  
4. Log savings (skipped layer/token counts) & quality.

**Outputs**
- FLOPs‑saved vs quality curves; latency reduction; failure cases (where unfreeze triggers often). Track skipped (token×layer×step) ratio.

---

## 9. P3 — Lightweight Learned Gate
**Supervision**
- From teacher traces, label at (t, i): **safe_to_freeze ∈ {0,1}** if freezing for next `K` steps would not change final task success (or proxy: final logits argmax/score unchanged).  
- Negative mining: include borderline cases where rule‑based failed.

**Features** (per token + small context):
`[cos, ΔL2, KL, entropy, margin, attn_sum, attn_entropy, step_index/num_steps, layer_band(one‑hot)]`.

**Model**
- Logistic regression or 2‑layer MLP (hidden 64) **<50k params**.  
- Outputs: `p_freeze`, and optional `freeze_length ∈ {1,2,3}` via 3‑way softmax.

**Training**
- Split by prompts; early‑stop on AUROC@dev; class‑imbalance by focal loss or pos_weight。  
- Export TorchScript to keep inference overhead negligible.

**Internal consistency checks**
- Final tokens consistency (Teacher vs wrapper with no skipping) should be 100%.
- Step‑wise KL / entropy trends must match Teacher traces.

---

## 10. P4 — Adaptive Step Scheduling
**Signal** (sequence‑level): moving average of **mean token KL** or **entropy slope** across steps.  
**Rule**: If below threshold for `r` consecutive steps → **increase scheduler stride** (skip 1 step); else revert to base stride.  
- Start with conservative `r=2`, stride=+1; cap total skipped steps ≤ 20%.

---

## 11. P5 — Combine & Evaluate
We evaluate **Teacher**, **RuleGate**, **LearnedGate**, **Adaptive**, and **Combined**.

**Metrics**
- **Latency** (ms/sequence), **Throughput** (seq/s @ batch=4), **GPU util** (optional via `nvidia-smi dmon`).  
- **Proxy FLOPs**: using `thop` / `fvcore` on executed ops; plus **count of skipped tokens×layers×steps**.  
- **Quality** per task (PPL, ROUGE‑L, EM/Acc).  
- **Stability**: % tokens ever frozen; average freeze length; unfreeze rate.

**Ablation grid**
- Thresholds: `τ∈{0.9,0.95,0.97}`, `ρ∈{0.05,0.10,0.15}`, `m∈{1,2,3}`, `K∈{1,2,3}`, `M∈{1,2,3}`.  
- Learned gate: with/without attn features; freeze length head on/off.  
- Adaptive steps: thresholds × stride.

---

## 12. Implementation Notes
- **Numerical safety**: clip tiny norms when computing ΔL2 ratio.  
- **Memory**: avoid storing all layers for all steps on big models — log **aggregates online** (mean/var/IQR); keep optional **sparse sampling** (e.g., every other step) for heavy runs.  
- **Overhead**: ensure gating cost < 3% runtime; prefer vectorized PyTorch ops; avoid Python loops on tokens.

---

## 13. Slurm Templates
**sbatch (teacher)** — `scripts/sbatch_teacher.sbatch`
```bash
#!/bin/bash
#SBATCH -J dcllm_teacher
#SBATCH -A <ACCOUNT>
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out

module load cuda/12.1 anaconda
source activate dcllm
export WANDB_MODE=online

srun bash scripts/run_teacher.sh reports
```

**run script (example)** — `scripts/run_teacher.sh`
```bash
python -m src.run \
  --mode teacher \
  --engine mock \
  --out_dir reports
```

**rule gate run** — `scripts/run_rule_gate.sh`
```bash
python -m src.run \
  --mode rule_gate \
  --engine mock \
  --out_dir reports \
  --tau 0.95 --rho 0.10 --m 2 --freeze_K 2 \
  --watchdog_min_cos 0.90 --watchdog_max_kl 0.02
```

---

## 14. Reporting & Visualization
**Auto‑generated figures** (per run):
1. **Cosine & ΔL2 heatmaps** (layers×steps).  
2. **Cosine histogram** with mean line.  
3. **Token stability** distribution (max consecutive stable steps).  
4. **FLOPs‑saved vs quality** curves for τ/ρ/K sweeps.  
5. **Latency vs quality** Pareto front comparing Teacher/Rule/Learned/Adaptive/Combined.  
6. **Failure case gallery**（where freezing hurts）with attention centrality overlays.

**Tables (CSV + W&B)**
- Per setting: {latency, throughput, ppl/rouge/em, skip_ratio, unfreeze_rate}.

**Final report** (`reports/final_<date>.pdf`)
- 6–8 pages with Abstract → Method → Experiments → Ablations → Discussion → Limitations → Appendix.

---

## 15. Risk Log & Guardrails
- **Accumulated error** → watchdog & periodic recompute (K/M), cap % of frozen tokens per step.  
- **Attention coupling** → avoid freezing **high centrality** tokens; or lower their τ.  
- **Distribution shift** → train learned gate on diverse prompts; include adversarial long‑range dependency cases.

---

## 16. Milestones & Owners
- **P0 (Day 0–0.5)**: Env + repo skeleton; engine wrapper compiling. *(Danny)*
- **P1 (Day 0.5–1)**: Teacher traces + 3 figures. *(Danny)*
- **P2 (Day 1–2)**: Rule‑based gating + FLOPs‑quality curve. *(Danny)*
- **P3 (Day 2–3)**: Learned gate (logistic→MLP). *(Danny)*
- **P4 (Day 3–4)**: Adaptive steps. *(Danny)*
- **P5 (Day 4–5)**: Combine + ablations + draft report. *(Danny; review: Celine/Zhenbang/Jai)*

---

## 17. Checklists
**Before each run**
- [ ] Commit hash recorded  
- [ ] Configs frozen (`.yaml` saved to run dir)  
- [ ] Seed fixed; dataset slice deterministic  
- [ ] W&B run name = `{mode}-{model}-{date}-{gitsha}`

**After each run**
- [ ] Pareto plots updated  
- [ ] Artifacts uploaded (traces if enabled)  
- [ ] Top 3 failure cases annotated

---

## 18. Config Examples
**configs/thresholds.yaml**
```yaml
rule_gate:
  cosine_tau: [0.90, 0.95, 0.97]
  delta_l2_rho: [0.05, 0.10, 0.15]
  consecutive_m: [1, 2, 3]
  freeze_K: [1, 2, 3]
  layer_recompute_M: [1, 2, 3]
  watchdog:
    min_cos: 0.90
    max_kl: 0.02
```

**configs/tasks.yaml**
```yaml
smoke:
  max_new_tokens: 128
  datasets:
    - name: wikitext
      subset: wikitext-2-raw-v1
      n_eval: 512
      metric: ppl
    - name: lambada_openai
      subset: en
      n_eval: 500
      metric: acc
    - name: gsm8k
      subset: main
      n_eval: 1000
      metric: em
```

---

## 19. Pseudocode Snippets
**Rule‑based per‑token freeze**
```python
state = engine.encode_prompt(prompt)
prev = None
for t in range(engine.num_steps()):
    if t == 0 or prev is None:
        out = engine.step(state, t)
    else:
        feats = compute_feats(out, prev)  # cos, ΔL2, KL, entropy, margin
        stable = (feats.cos>=tau) | (feats.dl2<=rho)
        counters = update_counters(stable)
        freeze_mask = (counters>=m)
        out = engine.step(state, t, compute_mask=~freeze_mask)
        out.hidden[freeze_mask] = prev.hidden[freeze_mask]
        out.logits[freeze_mask] = prev.logits[freeze_mask]
        # watchdog
        alert = (feats.cos<0.90) | (feats.kl>0.02)
        if alert.any():
            out = engine.step(state, t, compute_mask=None)
    prev = out
```

**Learned gate call**
```python
p, k = gate.predict(feats_batch)  # returns prob & length
freeze_mask = (p>0.5)
K = k.clamp(max=3)
```

---

## 20. Deliverables
- **Code**: clean PR with flags `--delta-freeze`, `--adaptive-steps`, `--router`.
- **W&B dashboard**: curated panels for all figures + Pareto front.  
- **Report PDF**: 6–8 pages with appendix; all plots reproducible via `scripts/make_report.py`.

---

## 21. Appendix — Glossary
- **cosine**: Cosine similarity between hidden at step t and t−1.
- **ΔL2 ratio**: `||h_t − h_{t-1}|| / (||h_{t-1}||+ε)`.
- **KL**: `KL(softmax z_t || softmax z_{t-1})`.
- **entropy, margin**: uncertainty and confidence gap from logits.
- **attention centrality**: total attn mass a token receives; higher → more influential.

