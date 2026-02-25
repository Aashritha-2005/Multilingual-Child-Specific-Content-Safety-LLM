"""
GEPA Adapter + Dataset Loader + Perplexity Client
For SemEval 2026 Task 9 — Multilingual Harmful Speech Classification
"""

import os
import json
import pandas as pd
import requests
import numpy as np

from typing import Dict, List, Any
from gepa import EvaluationBatch
from gepa.core.adapter import GEPAAdapter


# ================================================================
# 1. DATASET CONFIG
# ================================================================

CSV_PATH = "/Users/aashrithalakshmi/Downloads/GEPA/data/final_telugu_multilabel.csv"

LABELS = [
    "stereotype",
    "vilification",
    "dehumanization",
    "extreme_language",
    "lack_of_empathy",
    "invalidation",
]


# ================================================================
# 2. SEED PROMPT (RICH, DETAILED, READY FOR GEPA TO IMPROVE)
# ================================================================

SEED_PROMPT = """
You are an expert multilingual harmful-speech analyst.

Your goal is to classify social-media text (English + Telugu code-mixing) into SIX binary categories:

1. **stereotype**  
   - Overgeneralized assumptions about a group  
   - Examples:  
     - "All women are bad drivers"  
     - "వాళ్లు అంతా మోసగాళ్లు"

2. **vilification**  
   - Insulting, blaming, or demonizing language  
   - Examples:  
     - "These idiots ruin everything"  
     - "ఆ వాళ్లు చెడ్డ వాళ్లు"

3. **dehumanization**  
   - Denying human qualities (calling people animals, insects, objects)  
   - Examples:  
     - "They are cockroaches"  
     - "మనుషులే కారు వీళ్లు"

4. **extreme_language**  
   - Violent, hateful, or aggressive rhetoric  
   - Examples:  
     - "They should all suffer"  
     - "చంపేయాలి అనిపిస్తుంది"

5. **lack_of_empathy**  
   - Dismissing feelings or reducing suffering  
   - Examples:  
     - "Stop acting like a victim"  
     - "నీ సమస్య పెద్ద విషయం కాదు"

6. **invalidation**  
   - Denying experiences or identity  
   - Examples:  
     - "You’re overreacting, nothing happened"  
     - "అది ఏమీ కాదు, నువ్వు అలా అనవసరం"

-------------------------------------------------------------

### ❗ Output Format (Strict)
Return **ONLY** a JSON dictionary with EXACTLY these six keys:

{
  "stereotype": 0 or 1,
  "vilification": 0 or 1,
  "dehumanization": 0 or 1,
  "extreme_language": 0 or 1,
  "lack_of_empathy": 0 or 1,
  "invalidation": 0 or 1
}

No explanation. No text. No extra fields.
"""


# ================================================================
# 3. DATA LOADER
# ================================================================

def load_semeval(csv_path: str, sample_size=100):
    df = pd.read_csv(csv_path)
    required = ["text"] + LABELS
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    return df


# ================================================================
# 4. PERPLEXITY CLIENT
# ================================================================

class PerplexityClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.perplexity.ai/chat/completions"

    def generate(self, messages):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "sonar-pro",
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 350,
        }

        r = requests.post(self.url, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# ================================================================
# 5. GEPA ADAPTER
# ================================================================

class SemEvalGEPAAdapter(GEPAAdapter):
    """
    This adapter is FULLY compliant with GEPA 2025+.
    It:
    - Implements evaluate()
    - Implements make_reflective_dataset()
    - Returns EvaluationBatch
    - Passes trajectories for reflection
    """

    def __init__(self, csv_path: str, api_key: str, sample_size=80):
        self.df = load_semeval(csv_path, sample_size)
        self.client = PerplexityClient(api_key)
        self.labels = LABELS

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------
    def evaluate(self, batch, candidate, capture_traces=False):

        system_prompt = candidate["system_prompt"]

        outputs = []
        scores = []
        traces = [] if capture_traces else None

        for item in batch:
            text = item["text"]
            gt = {l: int(item[l]) for l in self.labels}

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Classify:\n{text}"}
            ]

            try:
                resp = self.client.generate(messages)
                pred = self._parse_prediction(resp)
            except:
                pred = {l: 0 for l in self.labels}

            outputs.append(pred)
            scores.append(self._instance_f1(pred, gt))

            if capture_traces:
                traces.append({
                    "text": text,
                    "raw_response": resp,
                    "prediction": pred,
                    "ground_truth": gt
                })

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=traces,
            objective_scores=None
        )

    # --------------------------------------------------------
    # REFLECTION
    # --------------------------------------------------------
    # --------------------------------------------------------
    # REFLECTION (FIXED VERSION)
    # --------------------------------------------------------
    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        """
        Create reflective dataset for prompt improvement.
        FIXED: Handles case where trajectories might be None.
        """
        
        # Guard against None trajectories
        if eval_batch.trajectories is None or len(eval_batch.trajectories) == 0:
            print("⚠️  WARNING: No trajectories available for reflection")
            print("   This usually means evaluate() was called with capture_traces=False")
            print("   GEPA internally will call evaluate() again with capture_traces=True")
            return {
                "system_prompt": []
            }
        
        records = []
        
        # Take first 5 trajectories for reflection
        for tr in eval_batch.trajectories[:5]:
            records.append({
                "text": tr["text"],
                "model_output": tr["raw_response"],
                "prediction": tr["prediction"],
                "ground_truth": tr["ground_truth"],
                "feedback": (
                    "Compare prediction with ground truth. "
                    "Explain what instructions should change to reduce errors."
                )
            })
        
        return {
            "system_prompt": records
        }

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------
    def _parse_prediction(self, resp):
        try:
            s = resp[resp.index("{") : resp.rindex("}") + 1]
            obj = json.loads(s)
            return {k: int(bool(obj.get(k, 0))) for k in self.labels}
        except:
            return {k: 0 for k in self.labels}

    def _instance_f1(self, pred, gt):
        return float(
            np.mean([1.0 if pred[l] == gt[l] else 0.0 for l in self.labels])
        )
