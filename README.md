# Delta-Compute Gating for Diffusion LLM

> Reduce end-to-end dLLM latency and FLOPs by computing only the necessary deltas across diffusion steps—via rule-based freezing (P2), a lightweight learned gate (P3), and adaptive step skipping (P4)—while formally controlling risk and preserving task quality.

This repository contains the research prototype for "Delta-Compute Gating," a collection of techniques to accelerate diffusion-based large language models (dLLMs) at inference time.

---

## Table of Contents
- [Core Concepts](#core-concepts)
- [Getting Started](#getting-started)
- [Repository Layout](#repository-layout)
- [Running Experiments](#running-experiments)
- [Evaluation Metrics](#evaluation-metrics)

---

## Core Concepts

The project is structured into several experimental phases, each building on the last:

- **P1 - Teacher:** A full-compute, unmodified run of the diffusion model. This phase logs performance metrics and per-step/layer features, which serve as a baseline and as training data for subsequent phases.
- **P2 - Rule-Based Gating:** Applies a set of hand-crafted rules to "freeze" (i.e., skip computation for) certain tokens or layers that are predicted to change little in a given diffusion step. This is the first step in saving real compute.
- **P3 - Learned Gate:** Replaces the hand-crafted rules with a small, lightweight neural network (an MLP or logistic regression model) trained to predict which tokens/layers can be safely frozen.
- **P4 - Adaptive Step Scheduling:** Instead of freezing parts of the computation within a step, this technique dynamically skips entire diffusion steps for a given sequence, further reducing latency.
- **P5 - Oracle & Baselines:** Establishes the theoretical upper bound (Oracle) and lower bound (Baselines) for performance. This helps contextualize the gains from phases P2, P3, and P4.

---

## Getting Started

### 1. Environment Setup
Create and activate the conda environment, then install the required packages.

```bash
# Create and activate the conda environment
conda create -y -n dcllm python=3.10
conda activate dcllm

# Install dependencies
pip install -r requirements.txt
```

### 2. Reproducibility
For reproducible results, it's recommended to set seeds at the start of your scripts.

```python
import torch, random, numpy as np

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Disable non-deterministic algorithms
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(False) # Set to True for full determinism if supported
```

---

## Repository Layout

The repository is organized to separate configuration, source code, scripts, and reports.

```
dcllm/
├─── README.md
├─── requirements.txt
├─── configs/              # Experiment, path, and task configurations
│    ├─── tasks.yaml
│    ├─── thresholds.yaml
│    └─── paths.yaml
├─── scripts/              # Shell scripts for running experiments
│    ├─── run_p1_teacher.sh
│    ├─── run_p2_rule_gate.sh
│    ├─── ...
│    └─── sbatch_p*_smoke_all.sbatch  # SLURM scripts for batch jobs
├─── src/                  # Main source code
│    ├─── run.py           # Main entry point for all experiments
│    ├─── engine/          # Core diffusion model execution logic
│    ├─── gating/          # Logic for P2, P3, and P4
│    ├─── eval/            # Evaluation tasks and scoring
│    ├─── probes/          # Metrics and feature extraction
│    └─── utils/           # Helper utilities
└─── reports/              # Output directory for logs, figures, and artifacts
```

---

## Running Experiments

The easiest way to run the Tier A smoke tests is to use the provided `sbatch` wrapper scripts. These scripts will launch jobs for the Wikitext-2, LAMBADA-open, and Tiny GSM8K datasets for each phase.

### Environment Variables
Before running, you may need to export the following environment variables to point to your data and the outputs of the P1 (Teacher) phase.

```bash
# Path to your datasets (e.g., /path/to/data)
export DATA_ROOT=/path/to/your/data

# Path to a completed P1 teacher run output directory
# This is needed for P2 (to get quantiles) and P3 (to get training features)
export P1_ROOT=/path/to/your/reports/P1/teacher/d2f/experiment_name
```

### Example Run Commands
The following commands will launch the full suite of smoke tests for each phase.

```bash
# P1: Run the teacher model to generate baselines and features
sbatch scripts/sbatch_p1_smoke_all.sbatch

# P2: Run the rule-based gating experiment (requires P1_ROOT to be set)
sbatch scripts/sbatch_p2_smoke_all.sbatch

# P3: Train and evaluate the learned gate (requires P1_ROOT to be set)
sbatch scripts/sbatch_p3_smoke_all.sbatch

# P4: Run the adaptive step scheduling experiment
sbatch scripts/sbatch_p4_smoke_all.sbatch

# P5: Run the oracle and baseline comparisons
sbatch scripts/sbatch_p5_smoke_all.sbatch
```

Each script runs a full evaluation and a smaller 5-sample trace run for detailed debugging. Outputs are saved to the `reports/` directory.

---

## Evaluation Metrics

The success of these methods is measured against a combination of performance, quality, and risk criteria.

- **Performance:**
  - **Latency (ms/seq):** Wall-clock time to generate a sequence.
  - **Throughput (seq/s):** Number of sequences processed per second.
  - **True FLOPs:** Floating-point operations, measured with a profiler or cost model.
- **Quality:**
  - **Task-specific metrics:** Perplexity (PPL) for Wikitext-2, Accuracy (Acc) for LAMBADA, and Exact Match (EM) for GSM8K.
  - **Consistency:** The fraction of generated tokens that match the output of the unmodified Teacher model.
- **Risk:**
  - **δ_frozen:** The false-positive rate over frozen/skipped items. This measures how often the model made a mistake *because* of a gating decision. The target is typically ≤1%.