"""
gepa_subtask3_expansion_fixed.py
==================================
GEPA Optimization with FORCED EXPANSION

KEY FIXES:
✓ Custom reflection prompt that FORCES expansion
✓ Proper reflection model (Claude Sonnet 3.5)
✓ Expansion-aware metric weighting
✓ Length-based validation

Usage:
    export OPENROUTER_API_KEY='your_key_here'
    python gepa_subtask3_expansion_fixed.py
"""



import os
from pathlib import Path
import sys
import json
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pandas as pd
from typing import Dict, List, Any
import requests
import time
from collections import Counter
import numpy as np

# GEPA imports
try:
    import gepa
    from gepa import EvaluationBatch
except ImportError:
    print("❌ GEPA not installed. Install with: pip install gepa")
    sys.exit(1)

# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = Path(__file__).parent / "data" / "final_telugu_multilabel.csv"

# CRITICAL: Use powerful reflection model for expansion
TASK_MODEL = "google/gemma-3-27b-it"
REFLECTION_MODEL = "deepseek/deepseek-r1-distill-llama-70b"  # MUST be powerful!

TEMPERATURE = 0.0
RANDOM_SEED = 42

# Exact label names from CSV
LABELS = [
    "stereotype",
    "vilification",
    "dehumanization",
    "extreme_language",
    "lack_of_empathy",
    "invalidation"
]

# Training config
TRAIN_SIZE = 40
VAL_SIZE = 15
MAX_METRIC_CALLS = 100

# Expansion targets
MIN_PROMPT_LENGTH = 8000  # characters
TARGET_EXPAND_MARKERS = 0  # All should be replaced

# ============================================================
# SEED PROMPT
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
# CUSTOM REFLECTION PROMPT TEMPLATE
# ============================================================

EXPANSION_REFLECTION_PROMPT = """You are a prompt engineering expert specializing in harmful speech classification for Telugu/English code-mixed text.EXPANSION

CRITICAL TASK: EXPAND THE PROMPT TO IMPROVE CLASSIFICATION ACCURACY

Current prompt length: {current_length} characters
Target length: 20,000-30,000 characters
[EXPAND] markers remaining: {expand_markers}

EXPANSION REQUIREMENTS (MANDATORY):
1. Replace EVERY [EXPAND] marker with actual content
2. Each label definition: 4-6 sentences with:
   - Academic definition
   - 5-8 Telugu/code-mixed examples
   - Linguistic markers (words, phrases, patterns)
   - Borderline cases to clarify boundaries
3. Section 11 (Calibration Examples): 15+ worked examples
   - 70% Telugu, 30% English
   - Show text → JSON → rationale
   - Include negative examples (all zeros)

ERROR ANALYSIS FROM VALIDATION:
{error_analysis}

MOST CONFUSED LABELS:
{confused_labels}

SECTIONS NEEDING MOST ATTENTION:
{underspecified_sections}

EXPANSION STRATEGY:
- For labels with high FP: Add counter-examples showing what's NOT that label
- For labels with high FN: Add more positive examples with clear markers
- For confused pairs: Add contrastive examples showing the difference
- ALL Telugu examples must use proper Telugu script (not transliteration)

OUTPUT REQUIREMENTS:
- Return the COMPLETE expanded prompt
- NO placeholders, NO [EXPAND] markers
- Minimum 20,000 characters
- Maintain original structure (sections, formatting)
- Keep JSON output format unchanged

Current prompt to expand:
{current_prompt}

EXPAND THIS PROMPT NOW:
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
        
        response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                verify=False   # ⚠️ TEMPORARY FIX
            )

        response.raise_for_status()
        data = response.json()
        
        return data["choices"][0]["message"]["content"]


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(csv_path: str) -> tuple:
    """Load and filter dataset."""
    print(f"📂 Loading dataset from {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"   Total samples: {len(df)}")
    
    # Filter zero-label examples
    df = df[df[LABELS].sum(axis=1) > 0].reset_index(drop=True)
    print(f"   After filtering: {len(df)} samples")
    
    # Shuffle
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Split
    train_df = df[:TRAIN_SIZE]
    val_df = df[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    
    # Convert to list of dicts
    train_data = train_df.to_dict('records')
    val_data = val_df.to_dict('records')
    
    print(f"✅ Split: {len(train_data)} train, {len(val_data)} val")
    
    return train_data, val_data


# ============================================================
# HELPERS
# ============================================================

def _parse_json_prediction(raw: str) -> Dict[str, int]:
    """Parse JSON from model response."""
    # Remove markdown
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


def _instance_accuracy(pred: Dict[str, int], gt: Dict[str, int]) -> float:
    """Compute instance accuracy."""
    correct = sum(pred[l] == gt[l] for l in LABELS)
    return correct / len(LABELS)


def _compute_batch_confusion(trajectories: List[Dict]) -> Dict:
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
            underspecified.append(f"{label}: FP={fp}, FN={fn}")
    
    return {
        "error_counts": dict(error_counts),
        "most_confused": most_confused,
        "underspecified": underspecified
    }


# ============================================================
# GEPA ADAPTER WITH CUSTOM REFLECTION
# ============================================================

class HarmfulSpeechAdapter(gepa.GEPAAdapter):
    """GEPA Adapter with forced expansion via custom reflection."""
    
    def __init__(self, api_key: str, model: str = TASK_MODEL, reflection_lm=None):
        self.client = OpenRouterClient(api_key=api_key, model=model)
        self.reflection_lm = reflection_lm

        
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
                pred = _parse_json_prediction(raw_resp)
            except requests.exceptions.HTTPError as e:
                print("❌ HTTP ERROR:")
                print(e.response.status_code)
                print(e.response.text)
                return EvaluationBatch(
                    outputs=[{label: 0 for label in LABELS}],
                    scores=[0.0],
                    trajectories=[] if capture_traces else None
                )


            
            outputs.append(pred)
            acc = _instance_accuracy(pred, gt)

            length_bonus = (len(system_prompt) / MIN_PROMPT_LENGTH) ** 0.5
            length_bonus = min(length_bonus, 1.0)

            final_score = acc * 0.4 + length_bonus * 0.6


            scores.append(final_score)

            
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
    
    def propose_new_texts(self, candidate, reflective_dataset, components_to_update):

        if not reflective_dataset:
            return {}

        new_texts = {}

        reflection_client = OpenRouterClient(
            api_key=os.environ["OPENROUTER_API_KEY"],
            model=self.reflection_lm
        )

        for component_name in components_to_update:

            records = reflective_dataset.get(component_name, [])
            if not records:
                continue

            current_prompt = records[0]["current_text"]

            reflection_prompt = EXPANSION_REFLECTION_PROMPT.format(
                current_length=len(current_prompt),
                expand_markers=current_prompt.count("[EXPAND"),
                error_analysis="...",
                confused_labels="...",
                underspecified_sections="...",
                current_prompt=current_prompt
            )

            MAX_REFLECTION_CHARS = 8000
            if len(reflection_prompt) > MAX_REFLECTION_CHARS:
                reflection_prompt = reflection_prompt[:MAX_REFLECTION_CHARS]

            print(f"\n🧠 Expanding component: {component_name}")
            print(f"   Current length: {len(current_prompt)}")

            try:
                expanded_prompt = reflection_client.complete(
                    [{"role": "user", "content": reflection_prompt}],
                    temperature=0.9,
                    max_tokens=3000
                )

            except requests.exceptions.HTTPError:
                print("⚠️ Reflection hit limit → retrying smaller")

                expanded_prompt = reflection_client.complete(
                    [{"role": "user", "content": reflection_prompt}],
                    temperature=0.5,
                    max_tokens=1200
                )

            except Exception as e:
                print(f"   ❌ Expansion failed: {e}")
                continue

            print(f"   ✅ Expanded length: {len(expanded_prompt)}")


            if len(expanded_prompt) <= len(current_prompt):
                print("⚠️ Reflection did not expand → retrying stronger")

                retry_prompt = reflection_prompt + "\n\nCRITICAL: Your response MUST be LONGER than the original prompt. Add detailed examples, rules, and explanations."

                try:
                    expanded_prompt = reflection_client.complete(
                        [{"role": "user", "content": retry_prompt}],
                        temperature=0.9,
                        max_tokens=2500
                    )
                except Exception as e:
                    print(f"❌ Retry failed: {e}")
                    continue

                print(f"🔁 Retry expanded length: {len(expanded_prompt)}")

                # 🚀 Only reject if STILL shorter
                if len(expanded_prompt) <= len(current_prompt):
                    print("❌ Still not longer → rejecting mutation")
                    continue



            new_texts[component_name] = expanded_prompt


        return new_texts


    
    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        """Create reflective dataset for GEPA."""
        if eval_batch.trajectories is None:
            return {comp: [] for comp in components_to_update}
        
        batch_analysis = _compute_batch_confusion(eval_batch.trajectories)
        
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
                    "confusion_signal": (
                        f"Most confused: {batch_analysis['most_confused']}"
                    ),
                    "underspecified_sections": batch_analysis["underspecified"],
                })
            
            reflective[comp] = records
        
        return reflective


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("GEPA OPTIMIZATION - FORCED EXPANSION MODE")
    print("=" * 60 + "\n")
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not set")
        print("\nGet your key from: https://openrouter.ai/keys")
        sys.exit(1)
    
    print(f"✅ API key found")
    print(f"🤖 Task model: {TASK_MODEL}")
    print(f"🧠 Reflection model: {REFLECTION_MODEL}")
    print(f"🎲 Random seed: {RANDOM_SEED}")
    print(f"\n⚠️  NOTE: Using Claude Sonnet for reflection (NOT free)")
    print(f"   Estimated cost: ~$0.50-2.00 for {MAX_METRIC_CALLS} calls")
    
    train_data, val_data = load_dataset(CSV_PATH)
    
    print(f"\n{'─' * 60}")
    print("INITIALIZING ADAPTER WITH CUSTOM REFLECTION")
    print(f"{'─' * 60}\n")
    
    adapter = HarmfulSpeechAdapter(
        api_key=api_key,
        model=TASK_MODEL,
        reflection_lm=REFLECTION_MODEL
    )

    
    print(f"✅ Adapter class: {adapter.__class__.__name__}")
    print(f"✅ Custom propose_new_texts: {hasattr(adapter, 'propose_new_texts')}")
    
    seed_candidate = {"system_prompt": SEED_PROMPT}
    
    print(f"\n📝 Seed prompt length: {len(SEED_PROMPT)} chars")
    print(f"📌 [EXPAND] markers: {SEED_PROMPT.count('[EXPAND')}")
    print(f"🎯 Target length: {MIN_PROMPT_LENGTH}+ chars")
    print(f"🎯 Target [EXPAND]: {TARGET_EXPAND_MARKERS}")
    
    print(f"\n{'─' * 60}")
    print("RUNNING GEPA OPTIMIZATION")
    print(f"{'─' * 60}\n")
    
    t0 = time.time()
    
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
    
    # Find longest/best prompt
    best = max(result.candidates, key=lambda c: len(c["system_prompt"]))["system_prompt"]
    
    print(f"\n📊 EXPANSION METRICS:")
    print(f"   Seed length: {len(SEED_PROMPT)} chars")
    print(f"   Final length: {len(best)} chars")
    print(f"   Growth: {len(best) - len(SEED_PROMPT):+d} chars ({100 * (len(best) - len(SEED_PROMPT)) / len(SEED_PROMPT):+.1f}%)")
    print(f"   [EXPAND] markers: {SEED_PROMPT.count('[EXPAND')} → {best.count('[EXPAND')}")
    print(f"   Target achieved: {'✅ YES' if len(best) >= MIN_PROMPT_LENGTH else '❌ NO'}")
    
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
            "expansion_ratio": len(best) / len(SEED_PROMPT),
            "expand_markers_removed": SEED_PROMPT.count('[EXPAND') - best.count('[EXPAND'),
            "val_scores": result.val_aggregate_scores,
        }, f, indent=2)
    
    print(f"\n💾 Results saved:")
    print(f"   - gepa_best_prompt.txt ({len(best)} chars)")
    print(f"   - gepa_results.json")
    print(f"\n📊 Validation scores: {result.val_aggregate_scores[:5]}...")
    print("\n✅ GEPA optimization complete!")


if __name__ == "__main__":
    main()