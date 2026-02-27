"""
run_gepa_gemini.py — FREE TIER VERSION
=======================================
Uses Google Gemini 1.5 Flash (permanent free tier) for reflection.
Rate limits: 15 RPM, 1500 RPD, 1M tokens/day.

SETUP INSTRUCTIONS
------------------
1. Go to https://aistudio.google.com/apikey
2. Click "Create API Key"
3. Copy the key (starts with AIza...)
4. Run:
     export GEMINI_API_KEY='AIza...'
     export PERPLEXITY_API_KEY='ppl_...'  # OR see alternative below

IMPORTANT: Gemini Flash is FREE but rate-limited. This script adds delays
between reflection calls to stay under 15 RPM.
"""

import os, sys, json, time, textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

import gepa
from gepa import EvaluationBatch


# ================================================================
# 1. GLOBAL CONFIG
# ================================================================

LANGUAGE = "tel"
CSV_PATH = Path(__file__).parent / "data" / "final_telugu_multilabel.csv"
SAMPLE_SIZE = 40  # Reduced from 60 to save API calls
VAL_SIZE = 15     # Reduced from 20
RANDOM_SEED = 42

# Task model — for eval calls (need something free here too)
# Options: (a) Keep Perplexity if you have credit
#          (b) Switch to Gemini Flash for eval too (slower but free)
USE_GEMINI_FOR_EVAL = False  # Set True if no Perplexity credit

TASK_MODEL = "openrouter/google/gemma-3-27b-it"
USE_GEMINI_FOR_EVAL = True


# Reflection LM — always Gemini Flash (free)
REFLECTION_LM = "openrouter/google/gemma-3-27b-it"

# CRITICAL: Rate limiting for free tier
# Gemini free tier = 15 RPM → one call every 4 seconds minimum
MIN_SECONDS_BETWEEN_REFLECTION_CALLS = 5.0

MAX_METRIC_CALLS = 200  # Reduced from 200 to finish faster

LABELS = [
    "stereotype",
    "vilification",
    "dehumanization",
    "extreme_language",
    "lack_of_empathy",
    "invalidation",
]

OUTPUT_DIR = Path(__file__).parent / "gepa_outputs"


# ================================================================
# 2. API KEY RESOLUTION
# ================================================================

def check_api_keys():
    """
    Verify required API keys are set.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        sys.exit(
            "\n✗ GEMINI_API_KEY not set.\n"
            "\n"
            "Get a free key:\n"
            "  1. Visit https://aistudio.google.com/apikey\n"
            "  2. Click 'Create API Key'\n"
            "  3. Copy the key (starts with AIza...)\n"
            "\n"
            "Then run:\n"
            "  export GEMINI_API_KEY='AIza...'\n"
            "\n"
            "Gemini 1.5 Flash is FREE (15 requests/min, 1.5M tokens/day).\n"
        )
    
    if not USE_GEMINI_FOR_EVAL:
        ppl_key = os.environ.get("PERPLEXITY_API_KEY", "").strip()
        if not ppl_key:
            print(
                "\n⚠ PERPLEXITY_API_KEY not set.\n"
                "  You can either:\n"
                "    (a) Set PERPLEXITY_API_KEY for fast eval calls, OR\n"
                "    (b) Edit run_gepa_gemini.py and set USE_GEMINI_FOR_EVAL=True\n"
                "        (slower but completely free)\n"
            )
            sys.exit(1)
    
    print("✓ API keys verified")
    return gemini_key, os.environ.get("PERPLEXITY_API_KEY", gemini_key)


# ================================================================
# 3. SEED PROMPT — same 13-section structure
# ================================================================

SEED_PROMPT = textwrap.dedent("""\
    ============================================================
    HARMFUL SPEECH CLASSIFIER — SYSTEM PROMPT
    Language context: {language} (code-mixed with English)
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
    • May be in {language}, English, or a mix of both scripts.
    • May contain informal spelling, slang, transliteration.
    • Length: typically 10-200 tokens.

    [EXPAND: Note any preprocessing assumptions.]

    ────────────────────────────────────────────────────────────
    SECTION 4 — OUTPUT SPECIFICATION
    ────────────────────────────────────────────────────────────
    Output ONLY a JSON object with exactly these six keys,
    each set to 0 or 1. No explanation. No extra text.

    {{
      "stereotype": 0 or 1,
      "vilification": 0 or 1,
      "dehumanization": 0 or 1,
      "extreme_language": 0 or 1,
      "lack_of_empathy": 0 or 1,
      "invalidation": 0 or 1
    }}

    ────────────────────────────────────────────────────────────
    SECTION 5 — LABEL DEFINITIONS (one block per label)
    ────────────────────────────────────────────────────────────

    --- 5a. stereotype ---
    Definition: [EXPAND: precise academic definition]
    Linguistic markers: [EXPAND: words/phrases]
    {language}-specific cues: [EXPAND]
    Borderline cases: [EXPAND]

    --- 5b. vilification ---
    Definition: [EXPAND]
    Linguistic markers: [EXPAND]
    {language}-specific cues: [EXPAND]
    Borderline cases: [EXPAND]

    --- 5c. dehumanization ---
    Definition: [EXPAND]
    Linguistic markers: [EXPAND]
    {language}-specific cues: [EXPAND]
    Borderline cases: [EXPAND]

    --- 5d. extreme_language ---
    Definition: [EXPAND]
    Linguistic markers: [EXPAND]
    {language}-specific cues: [EXPAND]
    Borderline cases: [EXPAND]

    --- 5e. lack_of_empathy ---
    Definition: [EXPAND]
    Linguistic markers: [EXPAND]
    {language}-specific cues: [EXPAND]
    Borderline cases: [EXPAND]

    --- 5f. invalidation ---
    Definition: [EXPAND]
    Linguistic markers: [EXPAND]
    {language}-specific cues: [EXPAND]
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
    Mix English + {language}. Include negative examples.]

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
""").format(language=LANGUAGE)


# ================================================================
# 4. DATA LOADER
# ================================================================

def load_dataset(csv_path, sample=SAMPLE_SIZE, val_n=VAL_SIZE, seed=RANDOM_SEED):
    df = pd.read_csv(csv_path)
    missing = [c for c in ["text"] + LABELS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df.head(sample + val_n)
    return df.iloc[:sample].to_dict("records"), df.iloc[sample:].to_dict("records")


# ================================================================
# 5. LLM CLIENTS
# ================================================================

class PerplexityClient:
    """HTTP client for Perplexity eval calls."""
    BASE = "https://api.perplexity.ai/chat/completions"

    def __init__(self, api_key: str, model: str = "sonar-pro"):
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.model = model

    def complete(self, messages, temperature=0.05):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 400,
        }
        r = requests.post(self.BASE, json=payload, headers=self.headers)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class GeminiClient:
    """HTTP client for Gemini eval calls (if USE_GEMINI_FOR_EVAL=True)."""
    BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash-latest"):
        self.api_key = api_key
        self.model = model

    def complete(self, messages, temperature=0.05):
        # Convert messages to Gemini format
        parts = [{"text": m["content"]} for m in messages if m["role"] != "system"]
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        if system_msg:
            parts.insert(0, {"text": system_msg})

        url = f"{self.BASE}/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 400,
            }
        }
        r = requests.post(url, json=payload)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]


# ================================================================
# 6. BATCH CONFUSION ANALYSIS
# ================================================================

def _compute_batch_confusion(traces):
    error_counts = Counter()
    pair_confusions = Counter()

    for t in traces:
        pred, gt = t["prediction"], t["ground_truth"]
        active_errors = []
        for l in LABELS:
            if pred[l] != gt[l]:
                direction = "FN" if gt[l] == 1 else "FP"
                error_counts[(l, direction)] += 1
                active_errors.append(l)
        for i in range(len(active_errors)):
            for j in range(i + 1, len(active_errors)):
                pair = tuple(sorted([active_errors[i], active_errors[j]]))
                pair_confusions[pair] += 1

    most_confused_pair = pair_confusions.most_common(1)[0][0] if pair_confusions else None

    section_map = {
        "stereotype": ("5a", "definition + markers"),
        "vilification": ("5b", "definition + markers"),
        "dehumanization": ("5c", "definition + markers"),
        "extreme_language": ("5d", "definition + markers"),
        "lack_of_empathy": ("5e", "definition + markers"),
        "invalidation": ("5f", "definition + markers"),
    }
    underspecified = []
    for (label, _), count in error_counts.most_common(4):
        sec_id, what = section_map[label]
        underspecified.append(f"Section {sec_id} ({label}): expand {what} — {count} errors")

    if most_confused_pair:
        underspecified.append(
            f"Section 9: add rule distinguishing {most_confused_pair[0]} vs {most_confused_pair[1]}"
        )
    if error_counts:
        underspecified.append("Section 11: add examples for labels with errors")

    return {
        "error_counts": dict(error_counts.most_common()),
        "most_confused_pair": most_confused_pair,
        "underspecified_sections": underspecified,
    }


# ================================================================
# 7. ADAPTER
# ================================================================

class HarmfulSpeechAdapter(gepa.GEPAAdapter):
    def __init__(self, api_key: str, use_gemini_for_eval: bool):
        if use_gemini_for_eval:
            self.client = GeminiClient(api_key)
        else:
            self.client = PerplexityClient(api_key)

    def evaluate(self, batch, candidate, capture_traces=False):
        system_prompt = candidate["system_prompt"]
        outputs, scores = [], []
        traces = [] if capture_traces else None

        for item in batch:
            text = item["text"]
            gt = {l: int(item[l]) for l in LABELS}

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Classify:\n{text}"},
            ]
            try:
                raw_resp = self.client.complete(messages)
                pred = _parse_json_prediction(raw_resp)
            except Exception as exc:
                raw_resp = f"<ERROR: {exc}>"
                pred = {l: 0 for l in LABELS}

            outputs.append(pred)
            scores.append(_instance_accuracy(pred, gt))

            if capture_traces:
                traces.append({
                    "input_text": text,
                    "raw_response": raw_resp,
                    "prediction": pred,
                    "ground_truth": gt,
                    "score": scores[-1],
                })

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=traces)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        if eval_batch.trajectories is None:
            return {comp: [] for comp in components_to_update}

        batch_analysis = _compute_batch_confusion(eval_batch.trajectories)

        reflective = {}
        for comp in components_to_update:
            records = []
            for traj in eval_batch.trajectories[:20]:
                pred, gt = traj["prediction"], traj["ground_truth"]

                per_label_errors = []
                for l in LABELS:
                    if pred[l] != gt[l]:
                        direction = "MISSED" if gt[l] == 1 else "FALSE ALARM"
                        per_label_errors.append(f"{l}: {direction}")

                # expand_directive = (
                #     "\n"
                #     "ABSOLUTE REQUIREMENT — NON-NEGOTIABLE:\n"
                #     "This task is PROMPT EXPANSION, not minimal optimization.\n"
                #     "\n"
                #     "You MUST:\n"
                #     "1. Replace EVERY [EXPAND] block with fully written content.\n"
                #     "2. Expand ALL sections even if model accuracy does NOT improve.\n"
                #     "3. Prefer OVER-EXPLANATION over brevity.\n"
                #     "4. Add redundancy, clarification, and examples deliberately.\n"
                #     "\n"
                #     "MANDATORY CONTENT REQUIREMENTS:\n"
                #     "• Each label definition: 4–6 sentences\n"
                #     "• Linguistic markers: 8–12 examples per label\n"
                #     "• Telugu / English / Hindi-specific cues explicitly separated\n"
                #     "• Sections 6–9: step-by-step, bullet-heavy rules\n"
                #     "• Section 11: MINIMUM 20 fully worked examples\n"
                #     "  – Include false positives, false negatives, and edge cases\n"
                #     "\n"
                #     "TARGET PROMPT LENGTH: 25,000–30,000 CHARACTERS.\n"
                #     "Current prompt is INCOMPLETE and UNDER-SPECIFIED.\n"
                #     "\n"
                #     "CRITICAL:\n"
                #     "• DO NOT keep ANY [EXPAND] markers.\n"
                #     "• DO NOT summarize.\n"
                #     "• DO NOT optimize for brevity.\n"
                #     "• EXPANSION IS MORE IMPORTANT THAN SCORE IMPROVEMENT.\n"
                # )

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


# ================================================================
# 8. HELPERS
# ================================================================

def _parse_json_prediction(raw):
    try:
        s = raw[raw.index("{"):raw.rindex("}") + 1]
        obj = json.loads(s)
        return {l: int(bool(obj.get(l, 0))) for l in LABELS}
    except Exception:
        return {l: 0 for l in LABELS}


def _instance_accuracy(pred, gt):
    return float(np.mean([1.0 if pred[l] == gt[l] else 0.0 for l in LABELS]))


def _save_results(result, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prefer the longest accepted prompt (expanded one)
    best = max(
        result.candidates,
        key=lambda c: len(c["system_prompt"])
    )["system_prompt"]

    (output_dir / "optimised_prompt.txt").write_text(best, encoding="utf-8")

    summary = {
        "best_candidate": result.best_candidate,
        "val_aggregate_scores": result.val_aggregate_scores,
        "language": LANGUAGE,
        "task_model": TASK_MODEL,
        "reflection_lm": REFLECTION_LM,
        "max_metric_calls": MAX_METRIC_CALLS,
        "seed_prompt_length": len(SEED_PROMPT),
        "optimised_prompt_length": len(best),
    }
    (output_dir / "gepa_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n✓ Results saved to {output_dir}/")
    print(f"  seed length      : {len(SEED_PROMPT):,} chars")
    print(f"  optimised length : {len(best):,} chars")


# ================================================================
# 9. MAIN
# ================================================================

def main():
    print("\n" + "=" * 60)
    print("  GEPA OPTIMIZATION — FREE TIER (Google Gemini Flash)")
    print("=" * 60 + "\n")

    # Check API keys
    print("[1/5] Checking API keys...")
    gemini_key, eval_key = check_api_keys()
    print(f"      Reflection LM: {REFLECTION_LM}")
    print(f"      Task model: {TASK_MODEL}")

    # Load data
    print(f"\n[2/5] Loading dataset from {CSV_PATH}...")
    trainset, valset = load_dataset(CSV_PATH)
    print(f"      trainset={len(trainset)}, valset={len(valset)}")

    # Initialize adapter
    print("\n[3/5] Initializing adapter...")
    adapter = HarmfulSpeechAdapter(api_key=eval_key, use_gemini_for_eval=USE_GEMINI_FOR_EVAL)

    # Seed
    seed_candidate = {"system_prompt": SEED_PROMPT}
    print(f"\n[4/5] Seed prompt: {len(SEED_PROMPT):,} chars")
    print(f"      [EXPAND] markers: {SEED_PROMPT.count('[EXPAND')}")

    # Run GEPA
    print(f"\n[5/5] Running GEPA...")
    print(f"      Budget: {MAX_METRIC_CALLS} eval calls")
    print(f"      ⚠ Rate limit: {MIN_SECONDS_BETWEEN_REFLECTION_CALLS}s between reflection calls (Gemini free tier)")
    print()

    t0 = time.time()
    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=REFLECTION_LM,
        max_metric_calls=MAX_METRIC_CALLS,
        display_progress_bar=True,
    )
    elapsed = time.time() - t0

    print(f"\n⏱ Optimization finished in {elapsed / 60:.1f} min")

    # Save
    _save_results(result, OUTPUT_DIR)

    # Print final prompt
    print("\n" + "=" * 70)
    print("FINAL OPTIMIZED PROMPT (first 500 chars)")
    print("=" * 70)
    print(result.best_candidate["system_prompt"][:500] + "...")
    print("=" * 70)
    print(f"\nVal scores: {result.val_aggregate_scores}")
    print(f"See full prompt: {OUTPUT_DIR}/optimised_prompt.txt")


if __name__ == "__main__":
    main()