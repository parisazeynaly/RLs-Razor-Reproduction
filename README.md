# CausalForgetting: Causal Discovery of the Forgetting Mechanism in LLM Fine-tuning

> **Paper extension of:** *"RL's Razor: Why On-Policy Reinforcement Learning Forgets Less"*
> **Contribution:** First data-driven causal graph arbitrating the KL vs. representation vs. on-policy debate in catastrophic forgetting.

---

## Overview

The RL's Razor paper establishes that forward KL divergence *predicts* catastrophic forgetting in LLM fine-tuning — but explicitly leaves the causal mechanism as an open question. Concurrent work disagrees on the root cause:

| Paper | Claimed Root Cause |
|---|---|
| Shenfeld et al. (RL's Razor) | KL divergence from base policy |
| "Retaining by Doing" | On-policy sampling specifically |
| Function Vector paper (ICLR 2025) | Representational shift in attention heads |

**This project runs causal discovery (FCI algorithm) on observational fine-tuning metrics to recover the underlying causal graph** — directly arbitrating this debate without assuming a causal structure in advance.

---

## Research Questions

1. Is forward KL divergence a **direct cause** of forgetting, or a **proxy** for a deeper mechanism?
2. Is representational shift (CKA) a **mediator** between distributional shift and forgetting, or an independent cause?
3. Does the on-policy/off-policy distinction introduce a **separate causal path** to forgetting, beyond its effect on KL?

---

## Method

### Step 1 — Reproduce ParityMNIST Sweep

Replicate the toy setting from RL's Razor (3-layer MLP, ParityMNIST + FashionMNIST) across a hyperparameter sweep, training with:
- GRPO (on-policy RL)
- GRPO + KL regularization
- SFT on distribution 1
- SFT on distribution 2
- SFT on oracle distribution

At each checkpoint, collect:

| Variable | Description |
|---|---|
| `fwd_kl` | KL(π₀ ‖ π) on new task distribution |
| `rev_kl` | KL(π ‖ π₀) on new task distribution |
| `cka` | CKA similarity to base model representations |
| `weight_l1` | L1 norm of weight change |
| `weight_l2` | L2 norm of weight change |
| `new_acc` | Accuracy on ParityMNIST (new task) |
| `prior_acc` | Accuracy on FashionMNIST (prior task) |
| `forgetting` | Drop in prior_acc relative to base model |
| `is_onpolicy` | Binary: 1 for GRPO, 0 for SFT |
| `lr` | Learning rate |
| `step` | Training step / checkpoint index |

Target: ~500 observations across all methods and hyperparameter settings.

### Step 2 — FCI Causal Discovery

Run the FCI (Fast Causal Inference) algorithm on the collected dataset using the `causal-learn` library. FCI is chosen over PC because it handles **latent confounders** — important here since we cannot observe all internal model states.

```
Variables: [fwd_kl, rev_kl, cka, weight_l1, new_acc, forgetting, is_onpolicy]
Independence test: Fisher-Z (continuous variables)
Significance level: α = 0.05
Output: PAG (Partial Ancestral Graph)
```

The PAG encodes:
- **→** : direct causal edge
- **↔** : common latent cause
- **o→** : either direct cause or common cause (ambiguous)

### Step 3 — Causal Mediation Analysis

Given the recovered PAG, perform mediation analysis to decompose:

```
Total effect of [is_onpolicy] on [forgetting]
  = Direct effect
  + Indirect effect through [fwd_kl]
  + Indirect effect through [cka]
```

This quantifies how much of RL's forgetting advantage is explained by KL minimization vs. representational preservation vs. other paths.

---

## Project Structure

```
causal-forgetting/
│
├── data/                        # Collected metrics from sweep
│   └── metrics.csv              # rows=checkpoints, cols=variables
│
├── experiments/
│   ├── parity_mnist.py          # ParityMNIST dataset + MLP definition
│   ├── train_sweep.py           # Full hyperparameter sweep (SFT + GRPO)
│   └── collect_metrics.py       # Checkpoint instrumentation & metric logging
│
├── causal/
│   ├── fci_discovery.py         # FCI algorithm on metrics.csv
│   ├── mediation.py             # Causal mediation analysis
│   └── visualize_pag.py         # PAG visualization
│
├── notebooks/
│   ├── 01_reproduce_rls_razor.ipynb    # Reproduce original paper figures
│   ├── 02_causal_discovery.ipynb       # FCI + PAG recovery
│   └── 03_mediation_analysis.ipynb     # Decompose causal paths
│
├── results/
│   ├── pag.png                  # Recovered causal graph
│   ├── mediation_table.csv      # Direct / indirect effect estimates
│   └── figures/                 # All paper figures
│
├── paper/
│   └── main.tex                 # Workshop paper (4 pages)
│
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/yourusername/causal-forgetting
cd causal-forgetting
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0
torchvision
numpy
pandas
scikit-learn
causal-learn          # FCI, PC, causal discovery
matplotlib
seaborn
tqdm
```

### Running on Colab / Kaggle

Each individual training run (single hyperparameter config) takes **~2-5 minutes** on a free Colab T4 GPU. The full sweep of ~100 configs takes ~4-6 hours total, which can be parallelized across sessions.

```python
# Quick start — run a single config
python experiments/train_sweep.py \
    --method grpo \
    --lr 1e-4 \
    --epochs 2 \
    --seed 42 \
    --output_dir data/runs/
```

---

## Expected Results

Based on the RL's Razor paper and FCI's properties, we expect to find one of three possible causal structures:

**Hypothesis A (supports RL's Razor):**
```
is_onpolicy → fwd_kl → forgetting
                ↑
              cka (mediator)
```

**Hypothesis B (supports "Retaining by Doing"):**
```
is_onpolicy → forgetting
is_onpolicy → fwd_kl
fwd_kl ⊥ forgetting | is_onpolicy
```

**Hypothesis C (supports Function Vector paper):**
```
is_onpolicy → cka → forgetting
is_onpolicy → fwd_kl
fwd_kl → cka
```

The recovered PAG will distinguish between these — or reveal a fourth structure none of the existing papers anticipated.

---


---

## Relation to Prior Work

This project does **not** propose a new training algorithm. The contribution is purely **explanatory** — providing the first causally-identified mechanism underlying the RL vs. SFT forgetting gap, resolving the conflict between three concurrent empirical papers that reach different conclusions using correlation-based analysis.

---

## Citation

If you use this work, please also cite the original RL's Razor paper:

```bibtex
@article{shenfeld2025rlsrazor,
  title={RL's Razor: Why On-Policy Reinforcement Learning Forgets Less},
  author={Shenfeld et al.},
  year={2025}
}
```

---

## Author

Sara Hashemi
MSc Data Science, University of Naples Federico II
