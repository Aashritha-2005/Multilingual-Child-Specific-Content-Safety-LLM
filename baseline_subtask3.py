"""
gepa_subtask3_fixed.py
======================
PHASE 2: GEPA Optimization for Subtask 3 (ALL FIXES APPLIED)

CRITICAL FIXES FROM SHIVANSH'S ISSUES:
✓ Issue 1: Exact label names (no underscore mismatches)
✓ Issue 2: Only valid GEPA parameters (no invalid args)
✓ Issue 3: Separate task vs reflection models
✓ Issue 4: Filter zero-label examples
✓ Issue 5: Correct adapter pattern (not standalone metric)
✓ Issue 6: Return EvaluationBatch (not Prediction)

CORRECT GEPA USAGE:
- Import: from gepa import EvaluationBatch
- Use: gepa.optimize() function
- Adapter pattern: evaluate() and make_reflective_dataset()
- NO separate metric function needed!

Usage:
    export OPENROUTER_API_KEY='your_key_here'
    python gepa_subtask3_fixed.py
"""

import os
from pathlib import Path
import sys
import json
import pandas as pd
from typing import Dict, List, Any
import requests
import time
from collections import Counter
import numpy as np

# GEPA imports - CRITICAL: Only these exist in gepa library!
try:
    import gepa
    from gepa import EvaluationBatch
except ImportError:
    print("❌ GEPA not installed. Install with: pip install gepa-ai")
    sys.exit(1)

# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = Path(__file__).parent / "data" / "final_telugu_multilabel.csv"

# CRITICAL FIX (Issue 3): Different models for task vs reflection
TASK_MODEL = "google/gemma-3-27b-it"  # For predictions
REFLECTION_MODEL = "anthropic/claude-3.5-sonnet"  # For prompt optimization

TEMPERATURE = 0.0
RANDOM_SEED = 42

# CRITICAL: Exact label names from CSV
LABELS = [
    "stereotype",
    "vilification",
    "dehumanization",
    "extreme_language",
    "lack_of_empathy",
    "invalidation"
]

# Training config
TRAIN_SIZE = 40  # Reduced for faster testing
VAL_SIZE = 15
MAX_METRIC_CALLS = 200  # Budget

# ============================================================
# SEED PROMPT (Same structure as run_gepa_gemini.py)
# ============================================================

SEED_PROMPT = """============================================================
HARMFUL SPEECH CLASSIFIER — SYSTEM PROMPT
Language context: tel (code-mixed with English)
============================================================

────────────────────────────────────────────────────────────
SECTION 1 — HIGH-LEVEL TASK DESCRIPTION
────────────────────────────────────────────────────────────
You are an expert multilingual harmful-speech classifier.
Your task is to analyse social-media text and assign binary
labels across six categories of harmful speech.

[EXPAND: Add 2-3 sentences on why this task matters for
online-safety research and content moderation.]

────────────────────────────────────────────────────────────
SECTION 2 — MOTIVATION AND RESEARCH CONTEXT
────────────────────────────────────────────────────────────
[EXPAND: Briefly describe the SemEval shared-task context,
the importance of multilingual detection, and how code-mixed
text (Telugu + English) adds difficulty.]

────────────────────────────────────────────────────────────
SECTION 3 — INPUT SPECIFICATION
────────────────────────────────────────────────────────────
Input: A single text string from social media.
• May be in tel, English, or a mix of both scripts.
• May contain informal spelling, slang, transliteration.
• Length: typically 10-200 tokens.

[EXPAND: Note any preprocessing assumptions.]

────────────────────────────────────────────────────────────
SECTION 4 — OUTPUT SPECIFICATION
────────────────────────────────────────────────────────────
Output ONLY a JSON object with exactly these six keys,
each set to 0 or 1. No explanation. No extra text.

{
  "stereotype": 0 or 1,
  "vilification": 0 or 1,
  "dehumanization": 0 or 1,
  "extreme_language": 0 or 1,
  "lack_of_empathy": 0 or 1,
  "invalidation": 0 or 1
}

────────────────────────────────────────────────────────────
SECTION 5 — LABEL DEFINITIONS (one block per label)
────────────────────────────────────────────────────────────

--- 5a. stereotype ---
Definition: [EXPAND: precise academic definition]
Linguistic markers: [EXPAND: words/phrases]
tel-specific cues: [EXPAND]
Borderline cases: [EXPAND]

--- 5b. vilification ---
Definition: [EXPAND]
Linguistic markers: [EXPAND]
tel-specific cues: [EXPAND]
Borderline cases: [EXPAND]

--- 5c. dehumanization ---
Definition: [EXPAND]
Linguistic markers: [EXPAND]
tel-specific cues: [EXPAND]
Borderline cases: [EXPAND]

--- 5d. extreme_language ---
Definition: [EXPAND]
Linguistic markers: [EXPAND]
tel-specific cues: [EXPAND]
Borderline cases: [EXPAND]

--- 5e. lack_of_empathy ---
Definition: [EXPAND]
Linguistic markers: [EXPAND]
tel-specific cues: [EXPAND]
Borderline cases: [EXPAND]

--- 5f. invalidation ---
Definition: [EXPAND]
Linguistic markers: [EXPAND]
tel-specific cues: [EXPAND]
Borderline cases: [EXPAND]

────────────────────────────────────────────────────────────
SECTION 6 — GENERAL CLASSIFICATION RULES
────────────────────────────────────────────────────────────
[EXPAND: 4-6 high-level rules that apply across all labels]

────────────────────────────────────────────────────────────
SECTION 7 — DECISION-TREE RULES
────────────────────────────────────────────────────────────
[EXPAND: Step-by-step decision procedure]

────────────────────────────────────────────────────────────
SECTION 8 — EDGE-CASE POLICY
────────────────────────────────────────────────────────────
[EXPAND: How to handle sarcasm, irony, quoting, etc.]

────────────────────────────────────────────────────────────
SECTION 9 — CROSS-LABEL CONFUSION RULES
────────────────────────────────────────────────────────────
[EXPAND: Distinguishing criteria for confused label pairs]

────────────────────────────────────────────────────────────
SECTION 10 — ANNOTATOR DO's AND DON'Ts
────────────────────────────────────────────────────────────
DO:  [EXPAND: 3-5 positive guidelines]
DON'T: [EXPAND: 3-5 common mistakes]

────────────────────────────────────────────────────────────
SECTION 11 — CALIBRATION EXAMPLES
────────────────────────────────────────────────────────────
[EXPAND: 10-15 worked examples with text, JSON output, rationale.
Mix English + tel. Include negative examples.]

────────────────────────────────────────────────────────────
SECTION 12 — CALIBRATION REMINDER
────────────────────────────────────────────────────────────
[EXPAND: Short note reminding model to re-read examples]

────────────────────────────────────────────────────────────
SECTION 13 — STRICT OUTPUT FORMAT
────────────────────────────────────────────────────────────
• Return ONLY the JSON object. Nothing else.
• All six keys must be present.
• Values must be exactly 0 or 1 (integers).
• No markdown. No explanation. No preamble.
============================================================
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
        
    def complete(self, prompt: str, temperature: float = TEMPERATURE) -> str:
        """Send prompt and return response."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"❌ API error: {e}")
            return ""


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(
    csv_path: Path,
    sample_size: int = TRAIN_SIZE,
    val_size: int = VAL_SIZE
) -> tuple:
    """
    Load and split dataset for GEPA.
    
    CRITICAL FIX (Issue 4): Filter zero-label examples.
    
    Returns:
        (train_data, val_data) as lists of dicts
    """
    print(f"📂 Loading dataset from {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"   Total samples: {len(df)}")
    
    # CRITICAL FIX: Filter zero-label examples
    df_before = len(df)
    df = df[df[LABELS].sum(axis=1) > 0].reset_index(drop=True)
    df_after = len(df)
    print(f"   ⚠️  Filtered {df_before - df_after} zero-label examples")
    print(f"   Remaining: {df_after} samples")
    
    # Shuffle
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Print label distribution
    print(f"\n📊 Label Distribution:")
    for label in LABELS:
        count = df[label].sum()
        pct = 100 * count / len(df)
        print(f"   {label:20s}: {count:3d} ({pct:5.1f}%)")
    
    # Split
    total_needed = sample_size + val_size
    if total_needed > len(df):
        print(f"⚠️  Requested {total_needed} samples, only {len(df)} available")
        sample_size = int(len(df) * 0.7)
        val_size = len(df) - sample_size
    
    train_df = df[:sample_size]
    val_df = df[sample_size:sample_size + val_size]
    
    # Convert to list of dicts
    train_data = train_df.to_dict("records")
    val_data = val_df.to_dict("records")
    
    print(f"\n✅ Split: {len(train_data)} train, {len(val_data)} val")
    
    return train_data, val_data


# ============================================================
# PREDICTION PARSING
# ============================================================

def _parse_json_prediction(response: str) -> Dict[str, int]:
    """Parse model response into label dictionary."""
    try:
        # Extract JSON from response
        s = response[response.index("{"):response.rindex("}") + 1]
        obj = json.loads(s)
        return {l: int(bool(obj.get(l, 0))) for l in LABELS}
    except Exception:
        return {l: 0 for l in LABELS}


def _instance_accuracy(pred: Dict[str, int], gt: Dict[str, int]) -> float:
    """Compute per-instance accuracy."""
    return float(np.mean([1.0 if pred[l] == gt[l] else 0.0 for l in LABELS]))


def _compute_batch_confusion(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze error patterns across batch."""
    error_counts = Counter()
    
    for trace in traces:
        pred = trace["prediction"]
        gt = trace["ground_truth"]
        
        for label in LABELS:
            if pred[label] == 1 and gt[label] == 0:
                error_counts[(label, "FP")] += 1
            elif pred[label] == 0 and gt[label] == 1:
                error_counts[(label, "FN")] += 1
    
    # Find most confused pair
    if error_counts:
        most_confused = error_counts.most_common(1)[0][0]
    else:
        most_confused = ("none", "none")
    
    # Identify underspecified sections
    underspecified = []
    for label in LABELS:
        fp = error_counts.get((label, "FP"), 0)
        fn = error_counts.get((label, "FN"), 0)
        if fp > 0 or fn > 0:
            underspecified.append(f"Section 5: {label} definition (FP={fp}, FN={fn})")
    
    return {
        "error_counts": dict(error_counts),
        "most_confused_pair": most_confused,
        "underspecified_sections": underspecified
    }


# ============================================================
# GEPA ADAPTER (CORRECT PATTERN)
# ============================================================

class HarmfulSpeechAdapter:
    """
    GEPA Adapter following the correct pattern from run_gepa_gemini.py
    
    CRITICAL: GEPA uses adapter pattern, NOT standalone metric functions!
    The adapter must implement:
    1. evaluate() - returns EvaluationBatch
    2. make_reflective_dataset() - returns dict of reflective records
    """
    
    def __init__(self, api_key: str, model: str = TASK_MODEL):
        self.client = OpenRouterClient(api_key=api_key, model=model)
        
    def evaluate(
        self,
        examples: List[Dict[str, Any]],
        candidate: Dict[str, Any],
        capture_traces: bool = False
    ) -> EvaluationBatch:
        """
        Evaluate prompt candidate on examples.
        
        Args:
            examples: List of data examples
            candidate: Dict with "system_prompt" key
            capture_traces: Whether to capture traces
            
        Returns:
            EvaluationBatch (NOT Prediction!)
        """
        system_prompt = candidate.get("system_prompt", "")
        
        scores = []
        outputs = []
        traces = [] if capture_traces else None
        
        for example in examples:
            # Use translated_te field for Telugu text
            text = example.get("translated_te", example.get("text", ""))
            
            # Build full prompt
            full_prompt = f"{system_prompt}\n\nText: {text}"
            
            # Get prediction
            response = self.client.complete(full_prompt)
            prediction = _parse_json_prediction(response)
            
            # Get ground truth (CRITICAL: use exact label names)
            ground_truth = {label: example[label] for label in LABELS}
            
            # Compute score
            score = _instance_accuracy(prediction, ground_truth)
            
            scores.append(score)
            outputs.append(prediction)
            
            if capture_traces:
                traces.append({
                    "input_text": text,
                    "raw_response": response,
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                    "score": score
                })
            
            time.sleep(0.3)  # Rate limiting
        
        # CRITICAL: Return EvaluationBatch, not Prediction!
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=traces
        )
    
    def make_reflective_dataset(
        self,
        candidate: Dict[str, Any],
        eval_batch: EvaluationBatch,
        components_to_update: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create reflective dataset for GEPA.
        
        Returns dict mapping component names to lists of reflective records.
        """
        if eval_batch.trajectories is None:
            return {comp: [] for comp in components_to_update}
        
        batch_analysis = _compute_batch_confusion(eval_batch.trajectories)
        
        reflective = {}
        for comp in components_to_update:
            records = []
            
            # Limit to 20 examples for efficiency
            for traj in eval_batch.trajectories[:20]:
                pred = traj["prediction"]
                gt = traj["ground_truth"]
                
                # Per-label errors
                per_label_errors = []
                for label in LABELS:
                    if pred[label] != gt[label]:
                        error_type = "MISSED" if gt[label] == 1 else "FALSE ALARM"
                        per_label_errors.append(f"{label}: {error_type}")
                
                # Expansion directive (same as run_gepa_gemini.py)
                expand_directive = (
                    "\n"
                    "CRITICAL LANGUAGE CONSTRAINT (NON-NEGOTIABLE):\n"
                    "The training dataset is PRIMARILY in Telugu (tel), written in Telugu script.\n"
                    "You MUST reflect this distribution in the optimized prompt.\n\n"
                    "SECTION 11 REQUIREMENTS (MANDATORY):\n"
                    "• At least 70% of all examples MUST be in Telugu script (తెలుగు).\n"
                    "• Telugu examples MUST NOT be English translations.\n"
                    "• Use natural social-media Telugu, including slang and informal grammar.\n"
                    "• Remaining examples may be English or Telugu-English code-mixed.\n\n"
                    "EXPANSION REQUIREMENTS:\n"
                    "• Replace EVERY [EXPAND] block with real content.\n"
                    "• Each label definition: 4–6 sentences.\n"
                    "• Linguistic markers: 8–12 examples per label, Telugu-first.\n"
                    "• Section 11: Minimum 25 examples total.\n"
                    "• Each Telugu example MUST be paired with correct JSON output.\n\n"
                    "TARGET LENGTH:\n"
                    "• Final prompt length: 30,000–40,000 characters.\n"
                    "• ZERO [EXPAND] placeholders may remain.\n"
                )
                
                records.append({
                    "component_name": comp,
                    "current_text": candidate.get(comp, ""),
                    "score": traj["score"],
                    "input": traj["input_text"],
                    "output": traj["raw_response"],
                    "per_label_errors": per_label_errors if per_label_errors else ["(all correct)"],
                    "confusion_signal": (
                        f"Batch errors: {batch_analysis['error_counts']}. "
                        f"Most confused: {batch_analysis['most_confused_pair']}."
                    ),
                    "expansion_directives": (
                        "YOU MUST EXPAND THE FOLLOWING SECTIONS IN DETAIL:\n"
                        + "\n".join(f"  • {d}" for d in batch_analysis["underspecified_sections"])
                        + "\nAdd explicit explanations, examples, and contrastive cases."
                    ),
                    "structural_completeness_directive": expand_directive,
                })
            
            reflective[comp] = records
        
        return reflective


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("GEPA OPTIMIZATION - SUBTASK 3 (CORRECTED)")
    print("=" * 60 + "\n")
    
    # Check API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not set")
        print("\nGet your key from: https://openrouter.ai/keys")
        sys.exit(1)
    
    print(f"✅ API key found")
    print(f"🤖 Task model: {TASK_MODEL}")
    print(f"🧠 Reflection model: {REFLECTION_MODEL}")
    print(f"🎲 Random seed: {RANDOM_SEED}")
    
    # Load data
    train_data, val_data = load_dataset(CSV_PATH)
    
    # Initialize adapter
    print(f"\n{'─' * 60}")
    print("INITIALIZING ADAPTER")
    print(f"{'─' * 60}\n")
    
    adapter = HarmfulSpeechAdapter(api_key=api_key, model=TASK_MODEL)
    
    # Seed candidate
    seed_candidate = {"system_prompt": SEED_PROMPT}
    
    print(f"📝 Seed prompt length: {len(SEED_PROMPT)} chars")
    print(f"📌 [EXPAND] markers: {SEED_PROMPT.count('[EXPAND')}")
    
    # Run GEPA
    print(f"\n{'─' * 60}")
    print("RUNNING GEPA OPTIMIZATION")
    print(f"{'─' * 60}\n")
    print(f"📊 Train size: {len(train_data)}")
    print(f"✅ Val size: {len(val_data)}")
    print(f"💰 Budget: {MAX_METRIC_CALLS} metric calls")
    print()
    
    t0 = time.time()
    
    # CRITICAL: Use gepa.optimize() function (not GEPA class!)
    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=train_data,
        valset=val_data,
        adapter=adapter,
        reflection_lm=REFLECTION_MODEL,
        max_metric_calls=MAX_METRIC_CALLS,
        display_progress_bar=True,
    )
    
    elapsed = time.time() - t0
    
    print(f"\n{'=' * 60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'=' * 60}\n")
    print(f"⏱️  Time: {elapsed / 60:.1f} minutes")
    
    # Get best candidate (prefer longest = most expanded)
    best = max(
        result.candidates,
        key=lambda c: len(c["system_prompt"])
    )["system_prompt"]
    
    print(f"📝 Optimized prompt length: {len(best)} chars")
    print(f"📈 Growth: {len(best) - len(SEED_PROMPT):+d} chars")
    print(f"📌 [EXPAND] markers remaining: {best.count('[EXPAND')}")
    
    # Save results
    output_dir = Path(__file__).parent
    
    with open(output_dir / "gepa_best_prompt.txt", "w", encoding="utf-8") as f:
        f.write(best)
    
    with open(output_dir / "gepa_results.json", "w") as f:
        json.dump({
            "task_model": TASK_MODEL,
            "reflection_model": REFLECTION_MODEL,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "max_metric_calls": MAX_METRIC_CALLS,
            "seed_prompt_length": len(SEED_PROMPT),
            "optimized_prompt_length": len(best),
            "val_scores": result.val_aggregate_scores
        }, f, indent=2)
    
    print(f"\n💾 Results saved:")
    print(f"   - gepa_best_prompt.txt")
    print(f"   - gepa_results.json")
    
    print(f"\n📊 Validation scores: {result.val_aggregate_scores[:5]}...")
    
    print("\n✅ GEPA optimization complete!")
    print("\n🔧 CORRECT GEPA USAGE:")
    print("   ✓ Import: from gepa import EvaluationBatch")
    print("   ✓ Function: gepa.optimize() (not GEPA class)")
    print("   ✓ Adapter pattern (evaluate + make_reflective_dataset)")
    print("   ✓ Return: EvaluationBatch (not Prediction)")
    print("   ✓ Exact label names")
    print("   ✓ Zero-label filtering")
    print("   ✓ Separate task/reflection models")


if __name__ == "__main__":
    main()

