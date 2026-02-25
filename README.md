# 🛡️ Multilingual Child-Specific Content Safety LLM  
### SemEval-2026 Task 9 — Subtask 3: Manifestation Identification

---

## 📌 Overview

This project investigates **multilingual polarization manifestation detection** in the context of **child-specific content safety**.

Built from **SemEval-2026 Task 9 (POLAR)**, the work focuses on:

> **Subtask 3: Manifestation Identification**  
> A **multi-label classification task** detecting how polarization is expressed in text.

The system explores translation-based normalization, prompt optimization, and structured reasoning pipelines using modern LLM tooling.

---

## 🎯 Task Definition

Given a multilingual text snippet, predict the presence (1) or absence (0) of:

- **Stereotype**
- **Vilification**
- **Dehumanization**
- **Extreme Language**
- **Lack of Empathy**
- **Invalidation**

⚠️ This is a **multi-label classification problem** — multiple manifestations may co-occur.

---

## 🌍 Languages

The SemEval dataset spans **22 languages**, including:

Amharic, Arabic, Bengali, Burmese, Chinese, English, German, Hausa, Hindi, Italian, Khmer, Nepali, Odia, Persian, Punjabi, Russian, Spanish, Swahili, Telugu, Turkish, Urdu.

---

## 🔄 Methodology

### 1️⃣ Dataset Translation Pipeline

To study low-resource language robustness and enable controlled evaluation:

- All samples were translated into **Telugu**
- Translation performed using **Gemma-based Google Translation**

**Motivation:**

✔ Normalize multilingual inputs  
✔ Simulate Indic safety pipelines  
✔ Evaluate translation-induced bias  

---

### 2️⃣ Baseline System

Initial experiments used:

- Seed prompts  
- Single-sentence inference  
- Direct label prediction  

Serving as a reference point for optimization.

---

### 3️⃣ Prompt Optimization

We applied:

- **GEPA (Guided Evolutionary Prompt Adaptation)**
- **GEPA + DSPy**

to evolve prompts for:

✔ Improved label discrimination  
✔ Multi-label consistency  
✔ Reduced hallucination  

---

### 4️⃣ Structured Classification (DSPy)

DSPy pipelines were introduced for:

- Declarative prompt structure  
- Modular reasoning  
- Label-wise optimization  

---

### 5️⃣ Error Analysis & Failure Mining

We performed:

- Label-segregated dataset evaluation  
- Misclassification tracking  
- Failure case clustering  

To identify:

✔ Ambiguity patterns  
✔ Prompt brittleness  
✔ Label confusion  

---

## ⚙️ Tech Stack

- **Python**
- **DSPy**
- **GEPA**
- **Google Gemma**
- **Transformers**
- **Pandas / NumPy**
- **Jupyter Notebooks**

---

## 📊 Experiments

| Experiment | Description |
|-----------|-------------|
| Baseline | Seed prompt classification |
| Translation Study | Multilingual → Telugu normalization |
| GEPA | Prompt evolution |
| GEPA + DSPy | Structured prompt optimization |
| Error Analysis | Label-wise failure investigation |

---

## 📈 Evaluation Focus

- Multi-label classification performance  
- Label-wise precision / recall  
- Translation robustness  
- Prompt stability  

---

## 🚀 Future Work

- Native multilingual modeling (without translation)
- Child-specific safety fine-tuning
- Cross-lingual generalization
- Bias & fairness evaluation
- Real-time moderation pipeline

---
