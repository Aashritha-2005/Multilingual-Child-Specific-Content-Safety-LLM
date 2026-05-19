# Multilingual Polarization Manifestation Identification in Telugu

**SemEval 2026 Task 9 — Subtask 3** | [Codabench Competition](https://www.codabench.org/competitions/10674/)

> Aashritha Lakshmi Mallampati · 2023A7PS0583P · BITS Pilani  
> Supervisors: Prof. Hari Babu | Prof. Yash Sinha

---

## Overview

This repository contains the **complete experimental programme** for SemEval 2026 Task 9, Subtask 3 — multi-label polarization manifestation identification in Telugu social media text.

The work is structured in two phases:

- **Phase 1** — Prompt engineering on large language models (root-level files)
- **Phase 2** — Encoder fine-tuning on the full Telugu training set (`phase2_finetuning/`)

The task requires classifying each post into six binary harmful speech categories:

| Label | Description |
|---|---|
| `stereotype` | Overgeneralised beliefs about a group |
| `vilification` | Hostile attack directed at a person or group |
| `dehumanization` | Comparing people to animals or objects |
| `extreme_language` | Radical, threatening, or inflammatory expressions |
| `lack_of_empathy` | Dismissing the suffering or rights of others |
| `invalidation` | Denying someone's identity or lived experience |

A single post can carry multiple labels simultaneously. Evaluation metric: **macro-averaged F1** across all six labels equally.

---

## Repository Structure

```
Multilingual-Child-Specific-Content-Safety-LLM/
│
├── phase1_results/                    # Stored outputs from Phase 1 experiments
├── probe_results/                     # Error analysis and failure mining outputs
│
│── Phase 1: Prompt Engineering (root-level) ─────────────────────────────────
│
├── phase1_multilabel.py               # S1: Llama-3 8B local prototype (Ollama)
├── phase2_stereotype.py               # S2: Stereotype trigger-based analysis
├── phase2_stereotype_binary.py        # S2: Stereotype binary detection variant
├── stereotype_dataset.py              # S2: Stereotype dataset utilities
├── baseline_subtask3.py               # S3: Zero-shot Gemma-3-27B baseline
├── gepa_subtask3.py                   # S4: GEPA evolutionary prompt optimisation
├── run_gepa.py                        # S4: GEPA runner script
├── run_gepa___.py                     # S4: GEPA alternate runner
├── run_gepa_program.py                # S7: Program-level structure evolution
├── dspy_subtask3.py                   # S5: DSPy BootstrapFewShot compilation
├── gepa_dspy_pipeline.py              # S6: GEPA + DSPy combined pipeline
├── unified_gepa_dspy_pipeline.py      # S6: Unified combined system (best Phase 1)
├── test_adapter.py                    # Testing utilities
├── test_baseline.py                   # Baseline testing utilities
├── test_gepa_imports.py               # Import verification
├── starter_pipeline.ipynb             # Starter notebook
├── gepa_best_prompt.txt               # Best evolved prompt (200-call GEPA run)
├── gepa_optimized_prompt.txt          # Optimised prompt output
├── gepa_results.json                  # GEPA run metrics
├── final_metrics.json                 # All Phase 1 experiment results
│
│── Phase 2: Encoder Fine-Tuning ──────────────────────────────────────────────
│
└── phase2_finetuning/
    ├── mps/                           # MPS backend outputs (Mac Apple Silicon)
    ├── output_v2/                     # Saved model checkpoints and OOF logits
    ├── data_preprocessing.py          # Data cleaning and label analysis utilities
    ├── cleaned_train.csv              # Preprocessed training data
    ├── baseline1.py                   # Ap.1-3: XLM-R/MuRIL, 12-label BCE
    ├── baseline2_with_split.py        # Ap.1-3: Baseline with upsampling + split
    ├── base_pipeline.py               # Ap.12: External augmentation pipeline
    ├── train_single.py                # Ap.1-3: Single model trainer (MuRIL/XLM-R)
    ├── train_pipeline.py              # Ap.1-3: Full ensemble pipeline
    ├── train_pipeline1.py             # Ap.6: XLM-R-Large + label smoothing
    ├── train_v2.py                    # Ap.4: Two-stage polarization gating
    ├── final_train.py                 # Ap.5/8: Corrected 6-label + SMASH workflow
    ├── ensemble_combined.py           # Ap.6: Weighted ensemble combination
    ├── ensemble_v2.py                 # Ap.8: Calibrated ensemble + full retrain
    ├── inference.py                   # Test inference on held-out data
    ├── inference_v2.py                # Inference variant with post-processing
    └── requirements.txt               # Phase 2 dependencies
```

---

## Dataset

Training data: **2,366 Telugu social media posts** from the SemEval 2026 Task 9 competition.

| Split | Samples | Polarized | Non-Polarized |
|---|---|---|---|
| Training | 2,366 | 1,274 | 1,092 |
| Test | 1,066 | 552 | 514 |

**Label distribution (training):**

| Label | Count | Prevalence |
|---|---|---|
| lack_of_empathy | 622 | 26.3% |
| invalidation | 539 | 22.8% |
| vilification | 536 | 22.7% |
| extreme_language | 318 | 13.4% |
| stereotype | 265 | 11.2% |
| **dehumanization** | **59** | **2.5%** ⚠️ |

> **Critical constraint:** Dehumanization has only 59 positive examples. With 5-fold CV, each fold trains on ~47 positives and validates on ~12. One wrong prediction shifts dehumanization F1 by 0.083.

---

## Phase 1: Prompt Engineering

Phase 1 implements a seven-stage progressive pipeline, each stage addressing limitations of the previous.

### Stage 1 — Local Small-Model Prototype
**File:** `phase1_multilabel.py`

Llama-3 8B served locally via Ollama (zero API cost). Hand-crafted 6-label prompt with definitions and three labelled examples each. Vilification scored only 0.178 F1 despite being the most prevalent label — conflated with extreme language under generic definitions.

**Result: Macro-F1 = 0.313**

---

### Stage 2 — Targeted Single-Label Analysis
**Files:** `phase2_stereotype.py`, `phase2_stereotype_binary.py`, `stereotype_dataset.py`

Isolated the stereotype label to measure prompt design impact directly. Simple definition prompt: 80/120 correct (66.7%). Redesigned with six explicit implicit-stereotype trigger patterns (gendered role framing, profession defaults, domestic role normalisation, tradition framing).

**Result: 66.7% → 94.2% (+27.5 percentage points from prompt design alone)**

---

### Stage 3 — Zero-Shot Large Model Baseline
**File:** `baseline_subtask3.py`

Gemma-3-27B-IT via OpenRouter API. TranslateGemma for Telugu→English. Structured JSON output per label. Zero-label filtering applied during optimisation.

| Label | F1 |
|---|---|
| Vilification | 1.000 |
| Dehumanization | 0.800 |
| Extreme Language | 0.800 |
| Lack of Empathy | 0.667 |
| Invalidation | 0.667 |
| Stereotype | 0.500 |
| **Overall Micro-F1** | **0.741** |

---

### Stage 4 — Evolutionary Prompt Optimisation (GEPA)
**Files:** `gepa_subtask3.py`, `run_gepa.py`, `run_gepa___.py`
**Outputs:** `gepa_best_prompt.txt`, `gepa_optimized_prompt.txt`, `gepa_results.json`

Seed prompt with 35 expansion placeholders. DeepSeek-R1-Distill-LLaMA-70B as reflection model. 200-call run expanded prompt from 4,763 → 27,087 characters (5.7×). Automatically added Telugu-specific examples, disambiguation rules, and inter-label conflict resolution.

**Result: Validation accuracy = 0.711**

---

### Stage 5 — Few-Shot Program Compilation (DSPy)
**File:** `dspy_subtask3.py`

ChainOfThought module required explicit reasoning before each prediction. BootstrapFewShot auto-selected demonstrations that led to correct predictions. No manual curation. Lack of empathy scored worst (0.40) — too few rare positive examples.

**Result: Validation score = 0.571**

---

### Stage 6 — Unified Combined System (GEPA + DSPy)
**Files:** `unified_gepa_dspy_pipeline.py`, `gepa_dspy_pipeline.py`

GEPA-evolved instructions as system prompt. BootstrapFewShot applied on top. Instruction optimisation reduces category confusion; demonstration optimisation provides reasoning trajectories. Outperformed both components individually.

**Result: Validation score = 0.633 — best combined Phase 1 result**

---

### Stage 7 — Program-Level Structure Evolution
**File:** `run_gepa_program.py`

Applied evolutionary optimiser to the full pipeline structure. Proposed a nine-stage architecture with dedicated sub-modules for language detection, cultural context, per-label assessment, and cross-label consistency. Search space too large for 200-call budget on small validation set.

**Result: Validation score = 0.000** (resource scaling issue, not a fundamental failure)

---

### Prompt Reasoning Benchmark — 12 Styles
Same Gemma-3-27B-IT backbone, 12 different reasoning structures, identical label definitions.

| Style | Macro-F1 |
|---|---|
| Robustness variation | **0.6052** |
| Active recall structure | 0.5817 |
| Structured subtasks | 0.5508 |
| Self-verification | 0.4821 |
| Step-by-step reasoning | 0.4203 |
| Draft-then-refine | 0.3895 |
| Explain-then-solve | 0.2571 |
| Metacognitive confidence | 0.2730 |
| Clarify if ambiguous | 0.1917 |

**41-point macro-F1 spread from reasoning structure alone — zero extra cost.**

---

## Phase 2: Encoder Fine-Tuning

All Phase 2 code is in `phase2_finetuning/`. Common setup across all approaches: 5-fold stratified CV, BCEWithLogitsLoss with per-label pos_weight, early stopping (patience=4), gradient clip=1.0, linear warmup.

### Approaches 1–3 — Joint Multi-Label Encoder Fine-Tuning
**Files:** `baseline1.py`, `baseline2_with_split.py`, `train_single.py`, `train_pipeline.py`

XLM-RoBERTa and MuRIL fine-tuned on all 12 label columns jointly. Weighted ensemble: MuRIL (0.85) + XLM-R (0.15). Best fine-tuning configuration overall.

**Result: OOF Macro-F1 = 0.473**

---

### Approach 4 — Two-Stage Polarization Gating Pipeline
**File:** `train_v2.py`

Stage A: binary polarization detector. Stage B: 6-label manifestation classifier gated on Stage A output. Focal loss alpha set to positive class frequency (inverted — dehumanization gradients vanished). Error propagation from Stage A compounded failures.

**Result: OOF Macro-F1 = 0.414 — worst fine-tuning result**

---

### Approach 5 — Corrected 6-Label BCE Classifier
**File:** `final_train.py`

Restricted to exactly the 6 subtask-3 labels. Fixed pos_weight formula. Kept non-polarized rows as negative anchors.

**Result: OOF Macro-F1 = 0.463**

---

### Approach 6 — Scaled Encoder with Label Smoothing (XLM-R-Large)
**File:** `train_pipeline1.py`

XLM-R-Large (560M parameters). Label smoothing pushed zero targets to 0.05 — 1,092 all-zero rows pulled all logits toward −2.94. Logit collapse. No gain from 4.5× model size increase.

**Result: OOF Macro-F1 = 0.460**

---

### Approach 7 — QLoRA Fine-Tuning (Qwen3-8B)

Qwen3-8B with 4-bit NF4 quantisation + LoRA rank-16. Model trained to generate structured label string as text output. Same approach used by top SemEval teams (27B models). Colab OOM errors prevented complete multi-fold evaluation. `max_new_tokens=60` caused output truncation.

**Result: Fold 1 Macro-F1 = 0.124** (truncation artifact — incomplete)

---

### Approach 8 — SMASH-Style Calibrated Ensemble + Full Retrain
**Files:** `ensemble_v2.py`, `final_train.py`

Replication of first-place SMASH pipeline: (1) 5-fold CV across multiple encoders + seeds → OOF logits, (2) grid-search per-model ensemble weights, (3) per-label threshold calibration on full OOF, (4) retrain 100% of data for test submission. OOF used only for threshold calibration, not as submission score.

**Result: OOF Macro-F1 = 0.447**

---

### Approaches 9–10 — Asymmetric Loss with Label Smoothing
Asymmetric Loss (gamma_neg=4, gamma_pos=1) + per-class weights [3.0, 1.5, 8.0, 3.0, 1.2, 1.5] + label smoothing. Smoothing created 6,552 negative-pulling gradient signals against 2,293 positive. Logit std collapsed to 0.02.

**Result: OOF Macro-F1 = ~0.155 — worst overall result**

---

### Approach 11 — Instruction-Augmented Fine-Tuning

English label definitions prepended to every Telugu input. 47-token preamble consumed ~25% of token budget, reducing effective Telugu representation.

**Result: OOF Macro-F1 = 0.428**

---

### Approach 12 — External Data Augmentation + Oversampling
**File:** `base_pipeline.py`

126 Telugu-translated harmful examples merged in. Dehumanization: 59→79 (+34%). Rare labels oversampled 3× within each fold after split (no leakage). Miscalibrated probability boost caused 181 dehumanization predictions against 59 actual.

**Result: OOF Macro-F1 = 0.356**

---

## Results Summary

| Approach | Method | Score |
|---|---|---|
| S3 | Zero-shot Gemma-3-27B + TranslateGemma | **0.741 Micro-F1** |
| Reasoning benchmark | Robustness variation style | **0.6052 Macro-F1** |
| S6 | GEPA + DSPy unified pipeline | 0.633 val |
| S4 | GEPA 200-call | 0.711 val acc |
| S5 | DSPy BootstrapFewShot | 0.571 val |
| Ap.1–3 | Encoder ensemble (MuRIL + XLM-R) | **0.473 OOF** |
| Ap.5 | Corrected 6-label BCE | 0.463 OOF |
| Ap.8 | SMASH calibrated ensemble | 0.447 OOF |
| Ap.11 | Instruction-augmented encoder | 0.428 OOF |
| Ap.4 | Two-stage gating pipeline | 0.414 OOF |
| Ap.12 | External augmentation | 0.356 OOF |
| Ap.7 | QLoRA Qwen3-8B (incomplete) | N/A |
| Ap.9–10 | ASL + label smoothing | ~0.155 OOF |
| TF-IDF in-sample ceiling | Logistic regression, same data | **0.703** |

---

## Key Findings

- **Prompt engineering outperformed all fine-tuning approaches.** Zero-shot 27B model (0.741 micro-F1) exceeded best fine-tuning OOF (0.473).
- **41-point macro-F1 spread from reasoning structure alone** — prompt style is a zero-cost optimisation lever.
- **All fine-tuning approaches plateau at 0.43–0.47 OOF** regardless of model size, loss function, or data strategy.
- **TF-IDF in-sample ceiling = 0.703** — the signal exists in the text. The 0.23-point generalisation gap is caused by dataset size, not model quality.
- **Label smoothing is definitively harmful** under extreme multi-label imbalance — it collapses logits when combined with many all-zero rows.
- **Full retrain on 100% of data is essential** for final submission — OOF scores underestimate test performance.

---

## Setup

### Phase 1 (Prompt Engineering)

```bash
git clone https://github.com/Aashritha-2005/Multilingual-Child-Specific-Content-Safety-LLM.git
cd Multilingual-Child-Specific-Content-Safety-LLM
pip install dspy-ai transformers pandas numpy openai requests

# Set OpenRouter API key
export OPENROUTER_API_KEY="your_key_here"

# Zero-shot baseline
python baseline_subtask3.py

# GEPA optimisation
python run_gepa.py

# DSPy compilation
python dspy_subtask3.py

# Unified GEPA + DSPy
python unified_gepa_dspy_pipeline.py
```

### Phase 2 (Fine-Tuning)

```bash
cd phase2_finetuning
pip install -r requirements.txt

# Best fine-tuning result — encoder ensemble (Colab GPU recommended)
python train_single.py --data_path ../data/tel_train.csv --model muril --device cuda
python train_single.py --data_path ../data/tel_train.csv --model xlmr  --device cuda

# SMASH-style calibrated ensemble
python ensemble_v2.py --data_path ../data/tel_train.csv --models muril,xlmr

# Inference on test data
python inference.py --test_path ../data/tel_test.csv
```

---

## Hardware Used

| Stage | Hardware |
|---|---|
| Phase 1 Stages 1–2 | Apple MacBook Pro M-series + Ollama (local) |
| Phase 1 Stages 3–7 | OpenRouter API — Gemma-3-27B-IT + DeepSeek-R1 |
| Phase 2 MPS runs | Apple MacBook Pro M-series (MPS backend — `mps/` folder) |
| Phase 2 GPU runs | Google Colab Tesla T4 (15.6 GB VRAM) |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Llama-3 8B via Ollama | Phase 1 Stage 1 local prototype |
| Gemma-3-27B-IT | Phase 1 Stages 3–7 classification |
| DeepSeek-R1-Distill-LLaMA-70B | GEPA reflection and error analysis |
| TranslateGemma | Telugu → English translation |
| GEPA | Evolutionary prompt optimisation |
| DSPy | Program compilation, BootstrapFewShot |
| XLM-RoBERTa | Phase 2 multilingual encoder |
| MuRIL | Phase 2 Indian-language encoder |
| PyTorch + HuggingFace Transformers | Phase 2 training framework |

---

## Future Directions

- Complete QLoRA evaluation with `max_new_tokens >= 100` and stable multi-fold checkpointing
- Cross-lingual transfer: train across all 22 SemEval languages before Telugu fine-tuning
- GEPA with 1,000-call budget on a full held-out validation set
- Pseudo-labelling targeting 500+ additional dehumanization positives
- Supervised contrastive loss to reduce vilification/dehumanization boundary confusion
- Native Telugu prompting without translation dependency

---

## References

- [SemEval 2026 Task 9 — arXiv:2604.06817](https://arxiv.org/abs/2604.06817)
- [Codabench Competition](https://www.codabench.org/competitions/10674/)
- [GEPA Framework](https://github.com/gepa-ai/gepa)
- [DSPy Framework](https://dspy.ai/)
- [TranslateGemma](https://blog.google/innovation-and-ai/technology/developers-tools/translategemma/)
- Khattab et al., DSPy: Compiling Declarative LLM Calls into Self-Improving Pipelines, arXiv:2310.03714
- Ridnik et al., Asymmetric Loss For Multi-Label Classification, ICCV 2021

---
