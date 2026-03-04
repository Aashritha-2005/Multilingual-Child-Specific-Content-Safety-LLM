"""
unified_gepa_dspy_pipeline.py
==============================
GEPA + DSPy Integration Pipeline

Pipeline:
1. GEPA prompt optimization
2. DSPy compilation using GEPA-optimized prompt
3. Final evaluation (GEPA + DSPy only)

Usage:
    export OPENROUTER_API_KEY='your_key_here'
    python unified_gepa_dspy_pipeline.py
"""

import os
from pathlib import Path
import sys
import json
import pandas as pd
from typing import Dict, List, Any, Tuple
import requests
import time
from collections import Counter
import numpy as np
import litellm

_original_completion = litellm.completion

def capped_completion(*args, **kwargs):
    kwargs["max_tokens"] = 256  # ← CRITICAL FIX
    kwargs["temperature"] = 0.0
    return _original_completion(*args, **kwargs)

litellm.completion = capped_completion


# Metrics
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# GEPA imports
try:
    import gepa
    from gepa import EvaluationBatch
except ImportError:
    print("❌ GEPA not installed. Install with: pip install gepa-ai")
    sys.exit(1)

# DSPy imports
try:
    import dspy
    from dspy import Signature, InputField, OutputField
except ImportError:
    print("❌ DSPy not installed. Install with: pip install dspy-ai")
    sys.exit(1)

# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = Path(__file__).parent / "data" / "final_telugu_multilabel.csv"

# Models
TASK_MODEL = "google/gemma-3-27b-it"
REFLECTION_MODEL = "openrouter/deepseek/deepseek-r1-distill-llama-70b"
TEMPERATURE = 0.0
RANDOM_SEED = 42

# Labels
LABELS = [
    "stereotype",
    "vilification",
    "dehumanization",
    "extreme_language",
    "lack_of_empathy",
    "invalidation"
]

# Training config
TRAIN_SIZE = 60
VAL_SIZE = 30
MAX_GEPA_CALLS = 50

# ============================================================
# SEED PROMPT
# ============================================================

SEED_PROMPT = """
HARMFUL SPEECH CLASSIFIER — SYSTEM PROMPT
Language context: Telugu + English (code-mixed)
============================================================

ROLE:
You are an expert multilingual harmful-speech classifier.
You analyse social-media text and assign binary labels
across six categories of harmful speech.

TASK:
Given a single text input, classify whether each label
is present (1) or absent (0).

LABELS:
1. stereotype
2. vilification
3. dehumanization
4. extreme_language
5. lack_of_empathy
6. invalidation

------------------------------------------------------------
OUTPUT FORMAT (STRICT)
------------------------------------------------------------
Return ONLY a valid JSON object:

{
  "stereotype": 0 or 1,
  "vilification": 0 or 1,
  "dehumanization": 0 or 1,
  "extreme_language": 0 or 1,
  "lack_of_empathy": 0 or 1,
  "invalidation": 0 or 1
}

Rules:
- No explanations
- No markdown
- No extra text
- All six keys must be present
- Values must be integers (0 or 1)

------------------------------------------------------------
INPUT CHARACTERISTICS
------------------------------------------------------------
- Text may be Telugu, English, or code-mixed
- May include slang, transliteration, spelling variation
- May contain sarcasm, irony, exaggeration
- May include quoted speech or reported speech

[EXPAND: Add preprocessing assumptions if needed]

------------------------------------------------------------
LABEL DEFINITIONS
------------------------------------------------------------

◆ stereotype
Definition:
Statements that generalise or attribute traits,
behaviours, or characteristics to a group.

Includes:
- Overgeneralizations about communities
- Group-based assumptions

Excludes:
- Neutral demographic descriptions

[EXPAND: Add linguistic markers]
[EXPAND: Add Telugu-specific cues]
[EXPAND: Add borderline examples]

------------------------------------------------------------

◆ vilification
Definition:
Content expressing hatred, contempt, or severe
negative judgement toward a group.

Includes:
- Insults targeting groups
- Calls for exclusion or hostility

Excludes:
- Criticism without hostility

[EXPAND: Add linguistic markers]
[EXPAND: Add Telugu-specific cues]
[EXPAND: Add borderline examples]

------------------------------------------------------------

◆ dehumanization
Definition:
Language portraying individuals/groups as
subhuman, animals, objects, or devoid of humanity.

Includes:
- Animalistic metaphors
- Denial of human qualities

Excludes:
- Non-harmful figurative speech

[EXPAND: Add linguistic markers]
[EXPAND: Add Telugu-specific cues]
[EXPAND: Add borderline examples]

------------------------------------------------------------

◆ extreme_language
Definition:
Highly inflammatory, violent, threatening,
or emotionally charged harmful language.

Includes:
- Threats
- Violent exaggerations
- Highly aggressive rhetoric

Excludes:
- Mild disagreement

[EXPAND: Add linguistic markers]
[EXPAND: Add Telugu-specific cues]
[EXPAND: Add borderline examples]

------------------------------------------------------------

◆ lack_of_empathy
Definition:
Dismissal, trivialisation, or mockery of suffering,
pain, tragedy, or emotional distress.

Includes:
- Minimizing harm
- Mocking victims

Excludes:
- Neutral disagreement

[EXPAND: Add linguistic markers]
[EXPAND: Add Telugu-specific cues]
[EXPAND: Add borderline examples]

------------------------------------------------------------

◆ invalidation
Definition:
Denying or undermining lived experiences,
identity, or legitimacy of feelings.

Includes:
- “That never happened”
- “Your feelings are fake”

Excludes:
- Respectful debate

[EXPAND: Add linguistic markers]
[EXPAND: Add Telugu-specific cues]
[EXPAND: Add borderline examples]

------------------------------------------------------------
GENERAL CLASSIFICATION RULES
------------------------------------------------------------

1. Labels are independent (multi-label possible)
2. Use semantic meaning, not just keywords
3. Consider tone, intent, and target
4. Account for sarcasm and irony carefully

[EXPAND: Add disambiguation rules]
[EXPAND: Add cross-label confusion guidance]

------------------------------------------------------------
EDGE-CASE POLICY
------------------------------------------------------------

Handle carefully:
- Sarcasm
- Quoted speech
- Hypotheticals
- Humour

[EXPAND: Add resolution strategy]

------------------------------------------------------------
CALIBRATION EXAMPLES
------------------------------------------------------------

[EXPAND: Add 5–10 worked examples]

------------------------------------------------------------
FINAL REMINDER
------------------------------------------------------------

Always:
✔ Return STRICT JSON
✔ No explanations
✔ No formatting deviations

"""

# ============================================================
# OPENROUTER CLIENT
# ============================================================

class OpenRouterClient:
    """OpenRouter API client."""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = TEMPERATURE,
        max_tokens: int = 400
    ) -> str:
        """Send messages and return response."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        response = requests.post(self.base_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        return data["choices"][0]["message"]["content"]

# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(csv_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Load and split dataset.
    
    Returns:
        (train_dicts, val_dicts)
    """
    print(f"📂 Loading dataset from {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"   Total samples: {len(df)}")
    
    # Filter zero-label examples
    df = df[df[LABELS].sum(axis=1) > 0].reset_index(drop=True)
    print(f"   After filtering zero-labels: {len(df)}")
    
    # Shuffle with fixed seed for reproducibility
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Split
    train_df = df[:TRAIN_SIZE]
    val_df = df[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    
    # Convert to dicts for GEPA
    train_data = train_df.to_dict('records')
    val_data = val_df.to_dict('records')
    
    print(f"✅ Split: {len(train_data)} train, {len(val_data)} val")
    
    return train_data, val_data

# ============================================================
# HELPERS
# ============================================================

def parse_json_prediction(raw: str) -> Dict[str, int]:
    """Parse JSON from model response."""
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    
    try:
        parsed = json.loads(text.strip())
        result = {}
        for label in LABELS:
            val = parsed.get(label, 0)
            result[label] = 1 if val in [1, "1", True, "true"] else 0
        return result
    except:
        return {label: 0 for label in LABELS}

def instance_accuracy(pred: Dict[str, int], gt: Dict[str, int]) -> float:
    """Compute instance accuracy."""
    correct = sum(pred[l] == gt[l] for l in LABELS)
    return correct / len(LABELS)

def compute_confusion_analysis(trajectories: List[Dict]) -> Dict:
    """Analyze errors across batch."""
    error_counts = Counter()
    
    for traj in trajectories:
        pred = traj["prediction"]
        gt = traj["ground_truth"]
        
        for label in LABELS:
            if pred[label] == 1 and gt[label] == 0:
                error_counts[(label, "FP")] += 1
            elif pred[label] == 0 and gt[label] == 1:
                error_counts[(label, "FN")] += 1
    
    most_confused = error_counts.most_common(2) if error_counts else []
    
    underspecified = []
    for label in LABELS:
        fp = error_counts.get((label, "FP"), 0)
        fn = error_counts.get((label, "FN"), 0)
        if fp > 0 or fn > 0:
            underspecified.append(f"Section 5: {label} (FP={fp}, FN={fn})")
    
    return {
        "error_counts": dict(error_counts),
        "most_confused": most_confused,
        "underspecified": underspecified
    }

def is_valid_prompt(prompt: str) -> bool:
    """Check if prompt is valid (not degenerate)."""
    if not prompt or len(prompt) < 100:
        return False
    if prompt.count("JSON") < 1:
        return False
    if not any(label in prompt for label in LABELS):
        return False
    return True

def get_candidate_score(c: Dict) -> float:
    """Robust score extraction from candidate."""
    return (
        c.get("aggregate_score")
        or c.get("score")
        or c.get("metrics", {}).get("aggregate")
        or 0
    )

# ============================================================
# METRICS COMPUTATION
# ============================================================

def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str]
) -> Dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels (n_samples, n_labels)
        y_pred: Predicted labels (n_samples, n_labels)
        label_names: List of label names
        
    Returns:
        Dictionary with per-label and aggregate metrics
    """
    results = {}
    
    # Per-label metrics
    for i, label in enumerate(label_names):
        y_true_label = y_true[:, i]
        y_pred_label = y_pred[:, i]
        
        results[label] = {
            "precision": precision_score(y_true_label, y_pred_label, zero_division=0),
            "recall": recall_score(y_true_label, y_pred_label, zero_division=0),
            "f1": f1_score(y_true_label, y_pred_label, zero_division=0),
            "accuracy": accuracy_score(y_true_label, y_pred_label),
            "support": int(y_true_label.sum())
        }
    
    # Macro-averaged metrics (primary metric)
    results["macro"] = {
        "precision": np.mean([results[l]["precision"] for l in label_names]),
        "recall": np.mean([results[l]["recall"] for l in label_names]),
        "f1": np.mean([results[l]["f1"] for l in label_names]),
        "accuracy": np.mean([results[l]["accuracy"] for l in label_names])
    }
    
    # Micro-averaged metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    results["micro"] = {
        "precision": precision_score(y_true_flat, y_pred_flat, zero_division=0),
        "recall": recall_score(y_true_flat, y_pred_flat, zero_division=0),
        "f1": f1_score(y_true_flat, y_pred_flat, zero_division=0),
        "accuracy": accuracy_score(y_true_flat, y_pred_flat)
    }
    
    # Subset accuracy (exact match)
    exact_match = np.all(y_true == y_pred, axis=1).mean()
    results["subset_accuracy"] = float(exact_match)
    
    return results

def print_metrics_table(metrics: Dict, title: str):
    """Pretty-print metrics table."""
    print(f"\n{'=' * 70}")
    print(f"{title:^70}")
    print(f"{'=' * 70}")
    
    # Per-label metrics
    print(f"\n{'Label':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 70)
    for label in LABELS:
        m = metrics[label]
        print(f"{label:<20} {m['precision']:>10.3f} {m['recall']:>10.3f} "
              f"{m['f1']:>10.3f} {m['support']:>10}")
    
    print("-" * 70)
    
    # Aggregate metrics
    print(f"\n{'Metric':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 70)
    
    macro = metrics["macro"]
    print(f"{'Macro-average':<20} {macro['precision']:>10.3f} "
          f"{macro['recall']:>10.3f} {macro['f1']:>10.3f}")
    
    # micro = metrics["micro"]
    # print(f"{'Micro-average':<20} {micro['precision']:>10.3f} "
    #       f"{micro['recall']:>10.3f} {micro['f1']:>10.3f}")
    
    # print(f"\n{'Subset Accuracy (Exact Match):':<40} {metrics['subset_accuracy']:>10.3f}")
    print("=" * 70)

# ============================================================
# GEPA ADAPTER
# ============================================================

class HarmfulSpeechAdapter(gepa.GEPAAdapter):
    """GEPA Adapter for harmful speech classification."""
    
    def __init__(self, api_key: str, model: str = TASK_MODEL):
        self.client = OpenRouterClient(api_key=api_key, model=model)
        
    def evaluate(self, batch, candidate, capture_traces=False):
        """Evaluate prompt candidate on batch."""
        system_prompt = candidate["system_prompt"]
        
        outputs = []
        scores = []
        traces = [] if capture_traces else None
        
        for item in batch:
            text = item.get("translated_te", item.get("text", ""))
            gt = {l: int(item[l]) for l in LABELS}
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Classify:\n{text}"},
            ]
            
            try:
                raw_resp = self.client.complete(messages, max_tokens=400)
                pred = parse_json_prediction(raw_resp)
            except Exception as exc:
                raw_resp = f"<ERROR: {exc}>"
                pred = {l: 0 for l in LABELS}
            
            outputs.append(pred)
            scores.append(instance_accuracy(pred, gt))
            
            if capture_traces:
                traces.append({
                    "input_text": text,
                    "raw_response": raw_resp,
                    "prediction": pred,
                    "ground_truth": gt,
                    "score": scores[-1],
                })
            
            time.sleep(0.5)
        
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=traces)
    
    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        """Create reflective dataset for GEPA."""
        if eval_batch.trajectories is None:
            return {comp: [] for comp in components_to_update}
        
        batch_analysis = compute_confusion_analysis(eval_batch.trajectories)
        
        reflective = {}
        for comp in components_to_update:
            records = []
            
            for traj in eval_batch.trajectories[:10]:
                pred = traj["prediction"]
                gt = traj["ground_truth"]
                
                per_label_errors = []
                for label in LABELS:
                    if pred[label] != gt[label]:
                        error_type = "MISSED" if gt[label] == 1 else "FALSE ALARM"
                        per_label_errors.append(f"{label}: {error_type}")
                
                records.append({
                    "component_name": comp,
                    "current_text": candidate.get(comp, ""),
                    "score": traj["score"],
                    "input": traj["input_text"],
                    "output": traj["raw_response"],
                    "per_label_errors": per_label_errors if per_label_errors else ["(all correct)"],
                    "confusion_signal": f"Most confused: {batch_analysis['most_confused']}",
                    "underspecified_sections": batch_analysis["underspecified"],
                })
            
            reflective[comp] = records
        
        return reflective

# ============================================================
# DSPY INTEGRATION
# ============================================================

class HarmfulSpeechSignature(Signature):
    """DSPy signature for harmful speech classification."""
    
    text = InputField(desc="Social media text in Telugu, English, or code-mixed")
    
    stereotype = OutputField(desc="1 if generalizations about groups, else 0")
    vilification = OutputField(desc="1 if hateful attacks on groups, else 0")
    dehumanization = OutputField(desc="1 if treating people as subhuman, else 0")
    extreme_language = OutputField(desc="1 if inflammatory/violent language, else 0")
    lack_of_empathy = OutputField(desc="1 if dismissing suffering, else 0")
    invalidation = OutputField(desc="1 if denying experiences, else 0")

class HarmfulSpeechClassifier(dspy.Module):
    """DSPy module with custom system prompt."""
    
    def __init__(self, system_prompt: str = None):
        super().__init__()
        
        # Use GEPA-optimized prompt as instructions
        instructions = system_prompt if system_prompt else SEED_PROMPT
        
        self.predictor = dspy.ChainOfThought(
            HarmfulSpeechSignature,
            instructions=instructions
        )
    
    def forward(self, text: str) -> dspy.Prediction:
        return self.predictor(text=text)

def dspy_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """DSPy metric for optimization."""
    correct = 0
    for label in LABELS:
        pred_val = getattr(pred, label, "0")
        pred_binary = 1 if str(pred_val).strip() in ["1", "True", "true"] else 0
        
        true_val = getattr(example, label, "0")
        true_binary = 1 if str(true_val).strip() in ["1", "True", "true"] else 0
        
        if pred_binary == true_binary:
            correct += 1
    
    return correct / len(LABELS)

# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def evaluate_dspy_compiled(
    compiled_classifier: dspy.Module,
    val_data: List[Dict]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate using GEPA + DSPy compiled classifier.
    """
    # print("\n" + "=" * 70)
    print("EVALUATING GEPA + DSPY COMPILED CLASSIFIER")
    print("=" * 70)
    
    y_true = []
    y_pred = []
    
    for i, item in enumerate(val_data):
        if (i + 1) % 10 == 0:
            print(f"   Processing {i + 1}/{len(val_data)}...")
        
        text = item.get("translated_te", item.get("text", ""))
        gt = [int(item[l]) for l in LABELS]
        
        try:
            prediction = compiled_classifier(text=text)
            
            pred = []
            for label in LABELS:
                val = getattr(prediction, label, "0")
                pred.append(1 if str(val).strip() in ["1", "True", "true"] else 0)
        except:
            pred = [0] * len(LABELS)
        
        y_true.append(gt)
        y_pred.append(pred)
        
        time.sleep(0.5)
    
    return np.array(y_true), np.array(y_pred)

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    # print("\n" + "=" * 70)
    print("GEPA + DSPY INTEGRATION PIPELINE")
    print("=" * 70)
    
    # Check API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not set")
        sys.exit(1)
    
    print(f"\n✅ API key found")
    print(f"🤖 Task model: {TASK_MODEL}")
    print(f"🧠 Reflection model: {REFLECTION_MODEL}")
    print(f"🎲 Random seed: {RANDOM_SEED}")
    print(f"🌡️  Temperature: {TEMPERATURE} (deterministic)")
    
    # Load data
    train_data, val_data = load_dataset(CSV_PATH)
    
    # Create output directory
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # ========================================
    # PHASE 1: GEPA OPTIMIZATION
    # ========================================
    
    # print("\n" + "=" * 70)
    print("PHASE 1: GEPA OPTIMIZATION")
    print("=" * 70)
    
    adapter = HarmfulSpeechAdapter(api_key=api_key, model=TASK_MODEL)
    seed_candidate = {"system_prompt": SEED_PROMPT}
    
    print(f"\n📝 Seed prompt: {len(SEED_PROMPT)} chars, {SEED_PROMPT.count('[EXPAND')} [EXPAND] markers")
    print(f"📊 Training with {len(train_data)} examples")
    print(f"✅ GEPA validation subset: {min(15, len(val_data))} examples")
    print(f"💰 Max metric calls: {MAX_GEPA_CALLS}")
    
    t0 = time.time()
    
    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=train_data,
        valset=val_data[:15],
        adapter=adapter,
        reflection_lm=REFLECTION_MODEL,
        max_metric_calls=MAX_GEPA_CALLS,
        display_progress_bar=True,
    )
    
    elapsed = time.time() - t0
    
    print(f"\n✅ GEPA optimization complete in {elapsed / 60:.1f} minutes")
    
    # Filter valid candidates
    valid_candidates = [c for c in result.candidates if is_valid_prompt(c.get("system_prompt", ""))]
    
    if not valid_candidates:
        print("⚠️  No valid candidates found, using best by length")
        valid_candidates = result.candidates
    
    print(f"   Valid candidates: {len(valid_candidates)}/{len(result.candidates)}")
    
    # Select best candidate by score
    best_candidate = max(valid_candidates, key=get_candidate_score)
    best_gepa_prompt = best_candidate["system_prompt"]
    best_score = get_candidate_score(best_candidate)
    
    print(f"\n📝 Best GEPA prompt:")
    print(f"   Length: {len(best_gepa_prompt)} chars")
    print(f"   [EXPAND] markers: {best_gepa_prompt.count('[EXPAND')}")
    # print(f"   Score: {best_score:.4f}")
    # print(f"   Growth: {len(best_gepa_prompt) - len(SEED_PROMPT):+d} chars ({100 * (len(best_gepa_prompt) - len(SEED_PROMPT)) / len(SEED_PROMPT):+.1f}%)")
    
    # Save GEPA prompt
    with open(output_dir / "gepa_optimized_prompt.txt", "w", encoding="utf-8") as f:
        f.write(best_gepa_prompt)
    
    print(f"\n💾 GEPA prompt saved to: {output_dir}/gepa_optimized_prompt.txt")
    
    # ========================================
    # PHASE 2: DSPY COMPILATION
    # ========================================
    
    # print("\n" + "=" * 70)
    print("PHASE 2: DSPY COMPILATION (USING GEPA PROMPT)")
    print("=" * 70)
    
    # Configure DSPy
    print(f"\n🔧 Configuring DSPy with GEPA-optimized prompt...")
    
    lm = dspy.LM(
        model=f"openrouter/{TASK_MODEL}",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        temperature=TEMPERATURE
    )
    dspy.configure(lm=lm)
    
    print(f"   ✅ DSPy LM configured: {TASK_MODEL}")
    print(f"   ✅ Temperature: {TEMPERATURE} (deterministic)")
    
    # Initialize classifier with GEPA prompt
    classifier = HarmfulSpeechClassifier(system_prompt=best_gepa_prompt)
    
    print(f"   ✅ Classifier initialized with GEPA prompt ({len(best_gepa_prompt)} chars)")
    
    # Convert train data to DSPy examples
    print(f"\n📚 Converting training data to DSPy examples...")
    
    train_examples = []
    for item in train_data:
        example = dspy.Example(
            text=item.get("translated_te", item.get("text", "")),
            **{label: str(item[label]) for label in LABELS}
        ).with_inputs("text")
        train_examples.append(example)
    
    print(f"   ✅ Converted {len(train_examples)} examples")
    
    # Compile with BootstrapFewShot
    print(f"\n⚙️  Compiling with BootstrapFewShot...")
    
    from dspy.teleprompt import BootstrapFewShot
    
    optimizer = BootstrapFewShot(
        metric=dspy_metric,
        max_bootstrapped_demos=8,
        max_labeled_demos=8
    )
    
    print(f"   Parameters:")
    print(f"   - max_bootstrapped_demos: 8")
    print(f"   - max_labeled_demos: 8")
    
    # Use deterministic compilation set (first 40 examples, no sampling)
    compilation_set = train_examples[:40]
    print(f"   - compilation_set: {len(compilation_set)} examples (deterministic)")
    
    t0_compile = time.time()
    
    compiled_classifier = optimizer.compile(
        classifier,
        trainset=compilation_set
    )
    
    elapsed_compile = time.time() - t0_compile
    
    print(f"\n✅ DSPy compilation complete in {elapsed_compile / 60:.1f} minutes")
    
    # ========================================
    # PHASE 3: FINAL EVALUATION
    # ========================================
    
    # print("\n" + "=" * 70)
    print("PHASE 3: FINAL EVALUATION (GEPA + DSPY)")
    print("=" * 70)
    
    print(f"\n📊 Evaluating on validation set ({len(val_data)} examples)...")
    
    y_true_final, y_pred_final = evaluate_dspy_compiled(compiled_classifier, val_data)
    
    # Compute comprehensive metrics
    print(f"\n📈 Computing comprehensive metrics...")
    
    metrics_final = compute_comprehensive_metrics(y_true_final, y_pred_final, LABELS)
    
    # Print metrics
    print_metrics_table(metrics_final, "FINAL METRICS: GEPA + DSPY")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    
    # print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    final_results = {
        "pipeline": "GEPA + DSPy",
        "task_model": TASK_MODEL,
        "reflection_model": REFLECTION_MODEL,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "gepa_calls": MAX_GEPA_CALLS,
        "seed_prompt_length": len(SEED_PROMPT),
        "optimized_prompt_length": len(best_gepa_prompt),
        "expand_markers_removed": SEED_PROMPT.count('[EXPAND') - best_gepa_prompt.count('[EXPAND'),
        "metrics": {
            "macro_f1": float(metrics_final["macro"]["f1"]),
            "macro_precision": float(metrics_final["macro"]["precision"]),
            "macro_recall": float(metrics_final["macro"]["recall"]),
            "micro_f1": float(metrics_final["micro"]["f1"]),
            "subset_accuracy": float(metrics_final["subset_accuracy"]),
            "per_label": {
                label: {
                    "precision": float(metrics_final[label]["precision"]),
                    "recall": float(metrics_final[label]["recall"]),
                    "f1": float(metrics_final[label]["f1"]),
                    "accuracy": float(metrics_final[label]["accuracy"]),
                    "support": int(metrics_final[label]["support"])
                }
                for label in LABELS
            }
        }
    }
    
    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Results saved:")
    print(f"   - {output_dir}/gepa_optimized_prompt.txt")
    print(f"   - {output_dir}/final_metrics.json")
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE - SUMMARY")
    print("=" * 70)
    
    print(f"\n🎯 FINAL RESULTS (GEPA + DSPY):")
    print(f"   Macro F1:        {metrics_final['macro']['f1']:.4f}")
    print(f"   Precision: {metrics_final['macro']['precision']:.4f}")
    print(f"   Recall:    {metrics_final['macro']['recall']:.4f}")
    # print(f"   Subset Accuracy: {metrics_final['subset_accuracy']:.4f}")
    
    # print(f"\n📝 PROMPT OPTIMIZATION:")
    # print(f"   Seed:      {len(SEED_PROMPT)} chars, {SEED_PROMPT.count('[EXPAND')} [EXPAND]")
    # print(f"   Optimized: {len(best_gepa_prompt)} chars, {best_gepa_prompt.count('[EXPAND')} [EXPAND]")
    # print(f"   Growth:    {len(best_gepa_prompt) - len(SEED_PROMPT):+d} chars ({100 * (len(best_gepa_prompt) - len(SEED_PROMPT)) / len(SEED_PROMPT):+.1f}%)")
    
    # print(f"\n⏱️  TIMING:")
    # print(f"   GEPA optimization: {elapsed / 60:.1f} minutes")
    # print(f"   DSPy compilation:  {elapsed_compile / 60:.1f} minutes")
    # print(f"   Total:             {(elapsed + elapsed_compile) / 60:.1f} minutes")
    
    print("\n✅ PIPELINE COMPLETE!")
    # print("=" * 70)

if __name__ == "__main__":
    main()

# """
# unified_gepa_dspy_pipeline.py
# ==============================
# UNIFIED PIPELINE: Baseline + GEPA + DSPy Integration

# Evaluates three approaches with comprehensive metrics:
# 1. Baseline (seed prompt, no optimization)
# 2. GEPA (optimized prompt)
# 3. GEPA + DSPy (optimized prompt + DSPy compilation)

# All variants evaluated with:
# - Precision, Recall, F1 (per-label and macro)
# - Accuracy
# - Confusion matrix analysis

# Usage:
#     export OPENROUTER_API_KEY='your_key_here'
#     python unified_gepa_dspy_pipeline.py
# """

# import os
# from pathlib import Path
# import sys
# import json
# from unittest import result
# import pandas as pd
# from typing import Dict, List, Any, Tuple
# import requests
# import time
# from collections import Counter
# import numpy as np
# import random
# import litellm

# _original_completion = litellm.completion

# def capped_completion(*args, **kwargs):
#     kwargs["max_tokens"] = 512
#     kwargs["temperature"] = 0.0
#     return _original_completion(*args, **kwargs)

# litellm.completion = capped_completion



# # Metrics
# from sklearn.metrics import (
#     precision_score,
#     recall_score,
#     f1_score,
#     accuracy_score,
#     classification_report
# )

# # GEPA imports
# try:
#     import gepa
#     from gepa import EvaluationBatch
# except ImportError:
#     print("❌ GEPA not installed. Install with: pip install gepa-ai")
#     sys.exit(1)

# # DSPy imports
# try:
#     import dspy
#     from dspy import Signature, InputField, OutputField
# except ImportError:
#     print("❌ DSPy not installed. Install with: pip install dspy-ai")
#     sys.exit(1)

# # ============================================================
# # CONFIGURATION
# # ============================================================

# CSV_PATH = Path(__file__).parent / "data" / "final_telugu_multilabel.csv"

# # Models
# TASK_MODEL = "google/gemma-3-27b-it"
# REFLECTION_MODEL = "openrouter/deepseek/deepseek-r1-distill-llama-70b"
# TEMPERATURE = 0.0
# RANDOM_SEED = 42

# # Labels
# LABELS = [
#     "stereotype",
#     "vilification",
#     "dehumanization",
#     "extreme_language",
#     "lack_of_empathy",
#     "invalidation"
# ]

# # Training config
# TRAIN_SIZE = 60
# VAL_SIZE = 20
# MAX_GEPA_CALLS = 40

# # ============================================================
# # SEED PROMPT
# # ============================================================

# SEED_PROMPT = """============================================================
# HARMFUL SPEECH CLASSIFIER — SYSTEM PROMPT
# Language context: tel (code-mixed with English)
# ============================================================

# ────────────────────────────────────────────────────────────
# SECTION 1 — HIGH-LEVEL TASK DESCRIPTION
# ────────────────────────────────────────────────────────────
# You are an expert multilingual harmful-speech classifier.
# Your task is to analyse social-media text and assign binary
# labels across six categories of harmful speech.

# [EXPAND: Add 2-3 sentences on why this task matters for
# online-safety research and content moderation.]

# ────────────────────────────────────────────────────────────
# SECTION 2 — MOTIVATION AND RESEARCH CONTEXT
# ────────────────────────────────────────────────────────────
# [EXPAND: Briefly describe the SemEval shared-task context,
# the importance of multilingual detection, and how code-mixed
# text (Telugu + English) adds difficulty.]

# ────────────────────────────────────────────────────────────
# SECTION 3 — INPUT SPECIFICATION
# ────────────────────────────────────────────────────────────
# Input: A single text string from social media.
# • May be in tel, English, or a mix of both scripts.
# • May contain informal spelling, slang, transliteration.
# • Length: typically 10-200 tokens.

# [EXPAND: Note any preprocessing assumptions.]

# ────────────────────────────────────────────────────────────
# SECTION 4 — OUTPUT SPECIFICATION
# ────────────────────────────────────────────────────────────
# Output ONLY a JSON object with exactly these six keys,
# each set to 0 or 1. No explanation. No extra text.

# {
#   "stereotype": 0 or 1,
#   "vilification": 0 or 1,
#   "dehumanization": 0 or 1,
#   "extreme_language": 0 or 1,
#   "lack_of_empathy": 0 or 1,
#   "invalidation": 0 or 1
# }

# ────────────────────────────────────────────────────────────
# SECTION 5 — LABEL DEFINITIONS (one block per label)
# ────────────────────────────────────────────────────────────

# --- 5a. stereotype ---
# Definition: [EXPAND: precise academic definition]
# Linguistic markers: [EXPAND: words/phrases]
# tel-specific cues: [EXPAND]
# Borderline cases: [EXPAND]

# --- 5b. vilification ---
# Definition: [EXPAND]
# Linguistic markers: [EXPAND]
# tel-specific cues: [EXPAND]
# Borderline cases: [EXPAND]

# --- 5c. dehumanization ---
# Definition: [EXPAND]
# Linguistic markers: [EXPAND]
# tel-specific cues: [EXPAND]
# Borderline cases: [EXPAND]

# --- 5d. extreme_language ---
# Definition: [EXPAND]
# Linguistic markers: [EXPAND]
# tel-specific cues: [EXPAND]
# Borderline cases: [EXPAND]

# --- 5e. lack_of_empathy ---
# Definition: [EXPAND]
# Linguistic markers: [EXPAND]
# tel-specific cues: [EXPAND]
# Borderline cases: [EXPAND]

# --- 5f. invalidation ---
# Definition: [EXPAND]
# Linguistic markers: [EXPAND]
# tel-specific cues: [EXPAND]
# Borderline cases: [EXPAND]

# ────────────────────────────────────────────────────────────
# SECTION 6 — GENERAL CLASSIFICATION RULES
# ────────────────────────────────────────────────────────────
# [EXPAND: 4-6 high-level rules that apply across all labels]

# ────────────────────────────────────────────────────────────
# SECTION 7 — DECISION-TREE RULES
# ────────────────────────────────────────────────────────────
# [EXPAND: Step-by-step decision procedure]

# ────────────────────────────────────────────────────────────
# SECTION 8 — EDGE-CASE POLICY
# ────────────────────────────────────────────────────────────
# [EXPAND: How to handle sarcasm, irony, quoting, etc.]

# ────────────────────────────────────────────────────────────
# SECTION 9 — CROSS-LABEL CONFUSION RULES
# ────────────────────────────────────────────────────────────
# [EXPAND: Distinguishing criteria for confused label pairs]

# ────────────────────────────────────────────────────────────
# SECTION 10 — ANNOTATOR DO's AND DON'Ts
# ────────────────────────────────────────────────────────────
# DO:  [EXPAND: 3-5 positive guidelines]
# DON'T: [EXPAND: 3-5 common mistakes]

# ────────────────────────────────────────────────────────────
# SECTION 11 — CALIBRATION EXAMPLES
# ────────────────────────────────────────────────────────────
# [EXPAND: 10-15 worked examples with text, JSON output, rationale.
# Mix English + tel. Include negative examples.]

# ────────────────────────────────────────────────────────────
# SECTION 12 — CALIBRATION REMINDER
# ────────────────────────────────────────────────────────────
# [EXPAND: Short note reminding model to re-read examples]

# ────────────────────────────────────────────────────────────
# SECTION 13 — STRICT OUTPUT FORMAT
# ────────────────────────────────────────────────────────────
# • Return ONLY the JSON object. Nothing else.
# • All six keys must be present.
# • Values must be exactly 0 or 1 (integers).
# • No markdown. No explanation. No preamble.
# ============================================================
# """

# # ============================================================
# # OPENROUTER CLIENT
# # ============================================================

# class OpenRouterClient:
#     """OpenRouter API client."""
    
#     def __init__(self, api_key: str, model: str):
#         self.api_key = api_key
#         self.model = model
#         self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
#     def complete(
#         self,
#         messages: List[Dict[str, str]],
#         temperature: float = TEMPERATURE,
#         max_tokens: int = 400
#     ) -> str:
#         """Send messages and return response."""
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json",
#         }
        
#         payload = {
#             "model": self.model,
#             "messages": messages,
#             "temperature": temperature,
#             "max_tokens": max_tokens,
#         }
        
#         response = requests.post(self.base_url, headers=headers, json=payload)
#         response.raise_for_status()
#         data = response.json()
        
#         return data["choices"][0]["message"]["content"]

# # ============================================================
# # DATA LOADING
# # ============================================================

# def load_dataset(csv_path: str) -> Tuple[List[Dict], List[Dict], pd.DataFrame]:
#     """
#     Load and split dataset.
    
#     Returns:
#         (train_dicts, val_dicts, val_df)
#     """
#     print(f"📂 Loading dataset from {csv_path}")
    
#     df = pd.read_csv(csv_path)
#     print(f"   Total samples: {len(df)}")
    
#     # Filter zero-label examples
#     df = df[df[LABELS].sum(axis=1) > 0].reset_index(drop=True)
#     print(f"   After filtering zero-labels: {len(df)}")
    
#     # Shuffle
#     df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
#     # Split
#     train_df = df[:TRAIN_SIZE]
#     val_df = df[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    
#     # Convert to dicts for GEPA
#     train_data = train_df.to_dict('records')
#     val_data = val_df.to_dict('records')
    
#     print(f"✅ Split: {len(train_data)} train, {len(val_data)} val")
    
#     return train_data, val_data, val_df

# # ============================================================
# # HELPERS
# # ============================================================

# def parse_json_prediction(raw: str) -> Dict[str, int]:
#     """Parse JSON from model response."""
#     text = raw.strip()
#     if "```json" in text:
#         text = text.split("```json")[1].split("```")[0]
#     elif "```" in text:
#         text = text.split("```")[1].split("```")[0]
    
#     try:
#         parsed = json.loads(text.strip())
#         result = {}
#         for label in LABELS:
#             val = parsed.get(label, 0)
#             result[label] = 1 if val in [1, "1", True, "true"] else 0
#         return result
#     except:
#         return {label: 0 for label in LABELS}
    
# def is_valid_prompt(text: str) -> bool:
#     """Reject degenerate GEPA mutations."""
#     if not text or len(text.strip()) == 0:
#         return False
#     if len(text.split()) < 150:
#         return False
#     if text.count("{") == 0:
#         return False
#     if "HARMFUL SPEECH CLASSIFIER" not in text:
#         return False

#     weird_ratio = sum(1 for c in text if ord(c) > 10000) / len(text)
#     if weird_ratio > 0.05:
#         return False

#     return True

# def instance_accuracy(pred: Dict[str, int], gt: Dict[str, int]) -> float:
#     """Compute instance accuracy."""
#     correct = sum(pred[l] == gt[l] for l in LABELS)
#     return correct / len(LABELS)

# def compute_confusion_analysis(trajectories: List[Dict]) -> Dict:
#     """Analyze errors across batch."""
#     error_counts = Counter()
    
#     for traj in trajectories:
#         pred = traj["prediction"]
#         gt = traj["ground_truth"]
        
#         for label in LABELS:
#             if pred[label] == 1 and gt[label] == 0:
#                 error_counts[(label, "FP")] += 1
#             elif pred[label] == 0 and gt[label] == 1:
#                 error_counts[(label, "FN")] += 1
    
#     most_confused = error_counts.most_common(2) if error_counts else []
    
#     underspecified = []
#     for label in LABELS:
#         fp = error_counts.get((label, "FP"), 0)
#         fn = error_counts.get((label, "FN"), 0)
#         if fp > 0 or fn > 0:
#             underspecified.append(f"Section 5: {label} (FP={fp}, FN={fn})")
    
#     return {
#         "error_counts": dict(error_counts),
#         "most_confused": most_confused,
#         "underspecified": underspecified
#     }

# # ============================================================
# # METRICS COMPUTATION
# # ============================================================

# def compute_comprehensive_metrics(
#     y_true: np.ndarray,
#     y_pred: np.ndarray,
#     label_names: List[str]
# ) -> Dict:
#     """
#     Compute comprehensive classification metrics.
    
#     Args:
#         y_true: Ground truth labels (n_samples, n_labels)
#         y_pred: Predicted labels (n_samples, n_labels)
#         label_names: List of label names
        
#     Returns:
#         Dictionary with per-label and aggregate metrics
#     """
#     results = {}
    
#     # Per-label metrics
#     for i, label in enumerate(label_names):
#         y_true_label = y_true[:, i]
#         y_pred_label = y_pred[:, i]
        
#         results[label] = {
#             "precision": precision_score(y_true_label, y_pred_label, zero_division=0),
#             "recall": recall_score(y_true_label, y_pred_label, zero_division=0),
#             "f1": f1_score(y_true_label, y_pred_label, zero_division=0),
#             "accuracy": accuracy_score(y_true_label, y_pred_label),
#             "support": int(y_true_label.sum())
#         }
    
#     # Macro-averaged metrics (primary metric)
#     results["macro"] = {
#         "precision": np.mean([results[l]["precision"] for l in label_names]),
#         "recall": np.mean([results[l]["recall"] for l in label_names]),
#         "f1": np.mean([results[l]["f1"] for l in label_names]),
#         "accuracy": np.mean([results[l]["accuracy"] for l in label_names])
#     }
    
#     # Micro-averaged metrics
#     y_true_flat = y_true.flatten()
#     y_pred_flat = y_pred.flatten()
#     results["micro"] = {
#         "precision": precision_score(y_true_flat, y_pred_flat, zero_division=0),
#         "recall": recall_score(y_true_flat, y_pred_flat, zero_division=0),
#         "f1": f1_score(y_true_flat, y_pred_flat, zero_division=0),
#         "accuracy": accuracy_score(y_true_flat, y_pred_flat)
#     }
    
#     # Subset accuracy (exact match)
#     exact_match = np.all(y_true == y_pred, axis=1).mean()
#     results["subset_accuracy"] = float(exact_match)
    
#     return results

# def print_metrics_table(metrics: Dict, title: str):
#     """Pretty-print metrics table."""
#     print(f"\n{'=' * 70}")
#     print(f"{title:^70}")
#     print(f"{'=' * 70}")
    
#     # Per-label metrics
#     print(f"\n{'Label':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
#     print("-" * 70)
#     for label in LABELS:
#         m = metrics[label]
#         print(f"{label:<20} {m['precision']:>10.3f} {m['recall']:>10.3f} "
#               f"{m['f1']:>10.3f} {m['support']:>10}")
    
#     print("-" * 70)
    
#     # Aggregate metrics
#     print(f"\n{'Metric':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
#     print("-" * 70)
    
#     macro = metrics["macro"]
#     print(f"{'Macro-average':<20} {macro['precision']:>10.3f} "
#           f"{macro['recall']:>10.3f} {macro['f1']:>10.3f}")
    
#     micro = metrics["micro"]
#     print(f"{'Micro-average':<20} {micro['precision']:>10.3f} "
#           f"{micro['recall']:>10.3f} {micro['f1']:>10.3f}")
    
#     print(f"\n{'Subset Accuracy (Exact Match):':<40} {metrics['subset_accuracy']:>10.3f}")
#     print("=" * 70)

# def macro_f1_single(pred: Dict[str, int], gt: Dict[str, int]) -> float:
#     """Macro-F1 computed at single-instance level."""
    
#     y_true = np.array([[gt[l] for l in LABELS]])
#     y_pred = np.array([[pred[l] for l in LABELS]])
    
#     f1s = []
#     for i in range(len(LABELS)):
#         f1s.append(
#             f1_score(
#                 y_true[:, i],
#                 y_pred[:, i],
#                 zero_division=0
#             )
#         )
    
#     return float(np.mean(f1s))


# # ============================================================
# # GEPA ADAPTER
# # ============================================================

# class HarmfulSpeechAdapter(gepa.GEPAAdapter):
#     """GEPA Adapter for harmful speech classification."""
    
#     def __init__(self, api_key: str, model: str = TASK_MODEL):
#         self.client = OpenRouterClient(api_key=api_key, model=model)
        
#     def evaluate(self, batch, candidate, capture_traces=False):
#         """Evaluate prompt candidate on batch."""
#         system_prompt = candidate["system_prompt"]
        
#         outputs = []
#         scores = []
#         traces = [] if capture_traces else None
        
#         for item in batch:
#             text = item.get("translated_te", item.get("text", ""))
#             gt = {l: int(item[l]) for l in LABELS}
            
#             messages = [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": f"Classify:\n{text}"},
#             ]
            
#             try:
#                 raw_resp = self.client.complete(messages, max_tokens=400)
#                 pred = parse_json_prediction(raw_resp)
#             except Exception as exc:
#                 raw_resp = f"<ERROR: {exc}>"
#                 pred = {l: 0 for l in LABELS}
            
#             outputs.append(pred)

            
#             scores.append(macro_f1_single(pred, gt))
            
#             if capture_traces:
#                 traces.append({
#                     "input_text": text,
#                     "raw_response": raw_resp,
#                     "prediction": pred,
#                     "ground_truth": gt,
#                     "score": scores[-1],
#                 })
            
#             time.sleep(0.5)
        
#         return EvaluationBatch(outputs=outputs, scores=scores, trajectories=traces)
    
    
#     def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
#         """Create reflective dataset for GEPA."""
        
#         if eval_batch.trajectories is None:
#             return {comp: [] for comp in components_to_update}

#         batch_analysis = compute_confusion_analysis(eval_batch.trajectories)

#         reflective = {}

#         for comp in components_to_update:
#             records = []

#             # ✅ Sort by score → worst first
#             sorted_trajs = sorted(
#                 eval_batch.trajectories,
#                 key=lambda t: t["score"]
#             )

#             # ✅ Use WORST predictions only
#             for traj in sorted_trajs[:5]:
#                 pred = traj["prediction"]
#                 gt = traj["ground_truth"]

#                 per_label_errors = []

#                 for label in LABELS:
#                     if pred[label] != gt[label]:
#                         error_type = (
#                             "MISSED" if gt[label] == 1 else "FALSE ALARM"
#                         )
#                         per_label_errors.append(f"{label}: {error_type}")

#                 records.append({
#                     "component_name": comp,
#                     "current_text": candidate.get(comp, ""),
#                     "score": traj["score"],
#                     "input": traj["input_text"],
#                     "output": json.dumps(pred),
#                     "per_label_errors": per_label_errors if per_label_errors else ["(all correct)"],
#                     "confusion_signal": (
#                         f"Frequent errors: {batch_analysis['most_confused']}. "
#                         f"Reduce FP/FN by clarifying label boundaries."
#                     ),
#                     "underspecified_sections": batch_analysis["underspecified"],
#                 })

#             reflective[comp] = records

#         return reflective


# # ============================================================
# # DSPY INTEGRATION
# # ============================================================

# class HarmfulSpeechSignature(Signature):
#     """DSPy signature for harmful speech classification."""
    
#     text = InputField(desc="Social media text in Telugu, English, or code-mixed")
    
#     stereotype = OutputField(desc="1 if generalizations about groups, else 0")
#     vilification = OutputField(desc="1 if hateful attacks on groups, else 0")
#     dehumanization = OutputField(desc="1 if treating people as subhuman, else 0")
#     extreme_language = OutputField(desc="1 if inflammatory/violent language, else 0")
#     lack_of_empathy = OutputField(desc="1 Text dismisses suffering / pain / tragedy, Not merely disagreement or criticism, else 0")
#     invalidation = OutputField(desc="1 if denying experiences, else 0")

# class HarmfulSpeechClassifier(dspy.Module):
#     """DSPy module with custom system prompt."""
    
#     def __init__(self, system_prompt: str = None):
#         super().__init__()
        
#         # CRITICAL: Use GEPA-optimized prompt as instructions
#         instructions = system_prompt if system_prompt else SEED_PROMPT
        
#         self.predictor = dspy.ChainOfThought(
#             HarmfulSpeechSignature,
#             instructions=instructions
#         )
    
#     def forward(self, text: str) -> dspy.Prediction:
#         return self.predictor(text=text)

# def dspy_metric(example, pred, trace=None):

#     y_true = []
#     y_pred = []

#     for label in LABELS:
#         true_val = getattr(example, label, "0")
#         pred_val = getattr(pred, label, "0")

#         y_true.append(1 if str(true_val) in ["1", "True", "true"] else 0)
#         y_pred.append(1 if str(pred_val) in ["1", "True", "true"] else 0)

#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)

#     f1s = [
#         f1_score([y_true[i]], [y_pred[i]], zero_division=0)
#         for i in range(len(LABELS))
#     ]

#     return float(np.mean(f1s))


# # ============================================================
# # EVALUATION FUNCTIONS
# # ============================================================

# def evaluate_baseline(
#     client: OpenRouterClient,
#     val_data: List[Dict],
#     system_prompt: str
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     --- BASELINE EVAL ---
#     Evaluate using seed prompt without optimization.
#     """
#     print("\n" + "=" * 70)
#     print("BASELINE EVALUATION (Seed Prompt)")
#     print("=" * 70)
    
#     y_true = []
#     y_pred = []
    
#     for i, item in enumerate(val_data):
#         if (i + 1) % 10 == 0:
#             print(f"   Processing {i + 1}/{len(val_data)}...")
        
#         text = item.get("translated_te", item.get("text", ""))
#         gt = [int(item[l]) for l in LABELS]
        
#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": f"Classify:\n{text}"}
#         ]
        
#         try:
#             response = client.complete(messages, max_tokens=400)
#             pred_dict = parse_json_prediction(response)
#             pred = [pred_dict[l] for l in LABELS]
#         except:
#             pred = [0] * len(LABELS)
        
#         y_true.append(gt)
#         y_pred.append(pred)
        
#         time.sleep(0.5)
    
#     return np.array(y_true), np.array(y_pred)

# def evaluate_gepa_prompt(
#     client: OpenRouterClient,
#     val_data: List[Dict],
#     gepa_prompt: str
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     --- GEPA EVAL ---
#     Evaluate using GEPA-optimized prompt.
#     """
#     print("\n" + "=" * 70)
#     print("GEPA EVALUATION (Optimized Prompt)")
#     print("=" * 70)
    
#     y_true = []
#     y_pred = []
    
#     for i, item in enumerate(val_data):
#         if (i + 1) % 10 == 0:
#             print(f"   Processing {i + 1}/{len(val_data)}...")
        
#         text = item.get("translated_te", item.get("text", ""))
#         gt = [int(item[l]) for l in LABELS]
        
#         messages = [
#             {"role": "system", "content": gepa_prompt},
#             {"role": "user", "content": f"Classify:\n{text}"}
#         ]
        
#         try:
#             response = client.complete(messages, max_tokens=400)
#             pred_dict = parse_json_prediction(response)
#             pred = [pred_dict[l] for l in LABELS]
#         except:
#             pred = [0] * len(LABELS)
        
#         y_true.append(gt)
#         y_pred.append(pred)
        
#         time.sleep(0.5)
    
#     return np.array(y_true), np.array(y_pred)

# def evaluate_dspy_compiled(
#     compiled_classifier: dspy.Module,
#     val_data: List[Dict]
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     --- DSPY EVAL ---
#     Evaluate using GEPA + DSPy compiled classifier.
#     """
#     print("\n" + "=" * 70)
#     print("DSPY EVALUATION (GEPA Prompt + DSPy Compilation)")
#     print("=" * 70)
    
#     y_true = []
#     y_pred = []
    
#     for i, item in enumerate(val_data):
#         if (i + 1) % 10 == 0:
#             print(f"   Processing {i + 1}/{len(val_data)}...")
        
#         text = item.get("translated_te", item.get("text", ""))
#         gt = [int(item[l]) for l in LABELS]
        
#         try:
#             prediction = compiled_classifier(text=text)
            
#             pred = []
#             for label in LABELS:
#                 val = getattr(prediction, label, "0")
#                 pred.append(1 if str(val).strip() in ["1", "True", "true"] else 0)
#         except:
#             pred = [0] * len(LABELS)
        
#         y_true.append(gt)
#         y_pred.append(pred)
        
#         time.sleep(0.5)
    
#     return np.array(y_true), np.array(y_pred)

# # ============================================================
# # MAIN PIPELINE
# # ============================================================

# def main():
#     print("\n" + "=" * 70)
#     print("UNIFIED PIPELINE: Baseline + GEPA + DSPy")
#     print("=" * 70)
    
#     # Check API key
#     api_key = os.environ.get("OPENROUTER_API_KEY")
#     if not api_key:
#         print("❌ OPENROUTER_API_KEY not set")
#         sys.exit(1)
    
#     print(f"\n✅ API key found")
#     print(f"🤖 Task model: {TASK_MODEL}")
#     print(f"🧠 Reflection model: {REFLECTION_MODEL}")
    
#     # Load data
#     train_data, val_data, val_df = load_dataset(CSV_PATH)
    
#     # Initialize client
#     client = OpenRouterClient(api_key=api_key, model=TASK_MODEL)
    
#     # ========================================
#     # PHASE 1: RUN GEPA OPTIMIZATION
#     # ========================================
    
#     print("\n" + "=" * 70)
#     print("PHASE 1: GEPA OPTIMIZATION")
#     print("=" * 70)
    
#     adapter = HarmfulSpeechAdapter(api_key=api_key, model=TASK_MODEL)
#     seed_candidate = {"system_prompt": SEED_PROMPT}
    
#     print(f"\n📝 Seed prompt: {len(SEED_PROMPT)} chars, {SEED_PROMPT.count('[EXPAND')} [EXPAND]")
#     print(f"📊 Training GEPA with {len(train_data)} examples...")
    
#     t0 = time.time()
    
#     result = gepa.optimize(
#         seed_candidate=seed_candidate,
#         trainset=train_data,
#         valset=val_data[:10],
#         adapter=adapter,
#         reflection_lm=REFLECTION_MODEL,
#         max_metric_calls=MAX_GEPA_CALLS,
#         display_progress_bar=True,
#     )


    
#     elapsed = time.time() - t0
#     valid_candidates = [
#         c for c in result.candidates
#         if "system_prompt" in c
#         and is_valid_prompt(c["system_prompt"])
#     ]


#     best_candidate = max(
#         valid_candidates if valid_candidates else result.candidates,
#         key=get_candidate_score
#     )


#     # Get best GEPA prompt
#     best_gepa_prompt = best_candidate["system_prompt"]


    
#     print(f"\n✅ GEPA complete in {elapsed / 60:.1f} minutes")
#     print(f"📝 Optimized prompt: {len(best_gepa_prompt)} chars, "
#           f"{best_gepa_prompt.count('[EXPAND')} [EXPAND]")
    
#     # Save GEPA prompt
#     output_dir = Path(__file__).parent / "outputs"
#     output_dir.mkdir(exist_ok=True)
    
#     with open(output_dir / "gepa_optimized_prompt.txt", "w") as f:
#         f.write(best_gepa_prompt)
    
#     # ========================================
#     # PHASE 2: BASELINE EVALUATION
#     # ========================================
    
#     print("\n" + "=" * 70)
#     print("PHASE 2: BASELINE EVALUATION")
#     print("=" * 70)
    
#     y_true_baseline, y_pred_baseline = evaluate_baseline(
#         client, val_data, SEED_PROMPT
#     )
    
#     metrics_baseline = compute_comprehensive_metrics(
#         y_true_baseline, y_pred_baseline, LABELS
#     )
    
#     print_metrics_table(metrics_baseline, "BASELINE METRICS (Seed Prompt)")
    
#     # ========================================
#     # PHASE 3: GEPA EVALUATION
#     # ========================================
    
#     print("\n" + "=" * 70)
#     print("PHASE 3: GEPA EVALUATION")
#     print("=" * 70)
    
#     y_true_gepa, y_pred_gepa = evaluate_gepa_prompt(
#         client, val_data, best_gepa_prompt
#     )
    
#     metrics_gepa = compute_comprehensive_metrics(
#         y_true_gepa, y_pred_gepa, LABELS
#     )
    
#     print_metrics_table(metrics_gepa, "GEPA METRICS (Optimized Prompt)")
    
#     # ========================================
#     # PHASE 4: DSPY COMPILATION
#     # ========================================
    
#     print("\n" + "=" * 70)
#     print("PHASE 4: DSPY COMPILATION")
#     print("=" * 70)
    
#     # Configure DSPy with GEPA-optimized prompt
#     lm = dspy.LM(
#         model=f"openrouter/{TASK_MODEL}",
#         api_key=api_key,
#         api_base="https://openrouter.ai/api/v1",
#         temperature=TEMPERATURE
#     )
#     dspy.configure(lm=lm)
    
#     # Create DSPy classifier with GEPA prompt
#     print(f"\n🔧 Initializing DSPy with GEPA-optimized prompt...")
#     classifier = HarmfulSpeechClassifier(system_prompt=best_gepa_prompt)
    
#     # Convert train data to DSPy examples
#     train_examples = []
#     for item in train_data:
#         example = dspy.Example(
#             text=item.get("translated_te", item.get("text", "")),
#             **{label: str(item[label]) for label in LABELS}
#         ).with_inputs("text")
#         train_examples.append(example)
    
#     # Compile with BootstrapFewShot
#     print(f"📚 Compiling with BootstrapFewShot...")
    
#     from dspy.teleprompt import BootstrapFewShot
    
#     optimizer = BootstrapFewShot(
#         metric=dspy_metric,
#         max_bootstrapped_demos=8,
#         max_labeled_demos=8
#     )
    
#     compilation_set = train_examples[:40]
    
#     compiled_classifier = optimizer.compile(
#         classifier,
#         trainset=compilation_set
#     )
    
#     print(f"✅ DSPy compilation complete")
    
#     # ========================================
#     # PHASE 5: DSPY EVALUATION
#     # ========================================
    
#     print("\n" + "=" * 70)
#     print("PHASE 5: DSPY EVALUATION")
#     print("=" * 70)
    
#     y_true_dspy, y_pred_dspy = evaluate_dspy_compiled(
#         compiled_classifier, val_data
#     )
    
#     metrics_dspy = compute_comprehensive_metrics(
#         y_true_dspy, y_pred_dspy, LABELS
#     )
    
#     print_metrics_table(metrics_dspy, "DSPY METRICS (GEPA + DSPy)")
    
#     # ========================================
#     # FINAL COMPARISON
#     # ========================================
    
#     print("\n" + "=" * 70)
#     print("FINAL COMPARISON - MACRO F1 SCORES")
#     print("=" * 70)
    
#     comparison = {
#         "Baseline (Seed Prompt)": metrics_baseline["macro"]["f1"],
#         "GEPA (Optimized Prompt)": metrics_gepa["macro"]["f1"],
#         "GEPA + DSPy (Compiled)": metrics_dspy["macro"]["f1"]
#     }
    
#     print(f"\n{'Approach':<30} {'Macro F1':>15} {'Improvement':>15}")
#     print("-" * 70)
    
#     baseline_f1 = comparison["Baseline (Seed Prompt)"]
    
#     for approach, f1 in comparison.items():
#         improvement = ((f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
#         print(f"{approach:<30} {f1:>15.4f} {improvement:>14.1f}%")
    
#     print("=" * 70)
    
#     # Save all results
#     results_summary = {
#         "baseline": {
#             "macro_f1": float(metrics_baseline["macro"]["f1"]),
#             "macro_precision": float(metrics_baseline["macro"]["precision"]),
#             "macro_recall": float(metrics_baseline["macro"]["recall"]),
#             "subset_accuracy": float(metrics_baseline["subset_accuracy"]),
#             "per_label": {l: metrics_baseline[l] for l in LABELS}
#         },
#         "gepa": {
#             "macro_f1": float(metrics_gepa["macro"]["f1"]),
#             "macro_precision": float(metrics_gepa["macro"]["precision"]),
#             "macro_recall": float(metrics_gepa["macro"]["recall"]),
#             "subset_accuracy": float(metrics_gepa["subset_accuracy"]),
#             "per_label": {l: metrics_gepa[l] for l in LABELS}
#         },
#         "gepa_dspy": {
#             "macro_f1": float(metrics_dspy["macro"]["f1"]),
#             "macro_precision": float(metrics_dspy["macro"]["precision"]),
#             "macro_recall": float(metrics_dspy["macro"]["recall"]),
#             "subset_accuracy": float(metrics_dspy["subset_accuracy"]),
#             "per_label": {l: metrics_dspy[l] for l in LABELS}
#         }
#     }
    
#     with open(output_dir / "unified_results.json", "w") as f:
#         json.dump(results_summary, f, indent=2)
    
#     print(f"\n💾 Results saved to:")
#     print(f"   - {output_dir}/gepa_optimized_prompt.txt")
#     print(f"   - {output_dir}/unified_results.json")
    
#     print("\n✅ UNIFIED PIPELINE COMPLETE!")

# def get_candidate_score(c):
#     """Robustly extract score from GEPA candidate."""
#     return (
#         c.get("aggregate_score")
#         or c.get("score")
#         or c.get("metrics", {}).get("aggregate")
#         or 0
#     )

    

# if __name__ == "__main__":
#     main()