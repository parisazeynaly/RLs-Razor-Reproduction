# Lightweight RL's Razor Reproduction for LLM Reasoning Fine-Tuning

This repository contains a lightweight research-oriented implementation inspired by **RL's Razor**, focused on comparing supervised fine-tuning and reinforcement learning fine-tuning for mathematical reasoning in small language models.

The project studies whether a simple **REINFORCE-style reinforcement learning** method can improve reasoning ability while causing less distributional shift from the original base model compared with **supervised fine-tuning (SFT)**.

---

## Research Question

**Does REINFORCE-style RL fine-tuning improve mathematical reasoning with less distributional shift than supervised fine-tuning in a lightweight LLM?**

In this project, the comparison is not between different model architectures. Instead, the same base model is adapted using different post-training strategies:

```text
Base Qwen2.5-0.5B-Instruct
├── Base model: no fine-tuning
├── SFT model: supervised fine-tuning on GSM8K
└── RL model: REINFORCE-style fine-tuning with correctness-based rewards
