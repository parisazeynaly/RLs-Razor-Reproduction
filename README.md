# RLs-Razor-Reproduction
# RL’s Razor: Reproduction Study  
*A complete reimplementation of “RL’s Razor: Why On-Policy Reinforcement Learning Forgets Less” (2025)*

---

## Overview

This repository contains a full **reproduction** of the experiments from the paper:

**RL’s Razor: Why On-Policy Reinforcement Learning Forgets Less**  
*(Anonymous Authors, 2025)*

The original paper demonstrates that:

- Reinforcement Learning (RL) fine-tuning forgets significantly less than Supervised Fine-Tuning (SFT)
- **KL divergence from the base model** is the key predictor of forgetting  
- On-policy RL methods naturally stay close to the base distribution  
- The KL–forgetting law holds both for **LLMs** and **toy settings (ParityMNIST)**

This repo reproduces all core results with clean, modular, and transparent code.

---

## 🚀 Goals of This Reproduction

This project aims to replicate:

###  LLM Experiments  
- SFT vs RL fine-tuning on:
  - Math reasoning
  - Science Q&A
  - Tool use  
- Forgetting evaluation on:
  - MMLU  
  - TruthfulQA  
  - HellaSwag  
  - Winogrande  
- Pareto frontiers: **New Task Accuracy vs Prior Task Performance**

###  KL Divergence Analysis  
- KL between base and fine-tuned models  
- KL as a predictor of catastrophic forgetting

### ParityMNIST Experiments  
- Single MLP model  
- Multiple SFT distributions  
- GRPO RL fine-tuning  
- Replication of the R² = ~0.96 KL–forgetting curve

###  Representation Shift (CKA)  
- Internal representation drift comparison:
  - SFT causes major drift  
  - RL preserves representations

---

##  Repository Structure

