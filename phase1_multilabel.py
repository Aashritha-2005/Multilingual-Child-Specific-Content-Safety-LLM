"""
PHASE 1 — Baseline Multilabel Classification (Telugu Dataset)
=============================================================
Evaluates harmful language classification using Claude API on Telugu text.
Ground-truth label columns: stereotype, vilification, dehumanization,
                            extreme_language, lack_of_empathy, invalidation
"""

import os
import re
# import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import requests

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATA_PATH        = "data/final_telugu_multilabel.csv"   # update path if needed
OUTPUT_DIR       = "phase1_results"
RESULTS_CSV      = os.path.join(OUTPUT_DIR, "phase1_predictions.csv")
METRICS_CSV      = os.path.join(OUTPUT_DIR, "phase1_metrics.csv")
# MODEL            = "claude-opus-4-6"               # change to claude-haiku-4-5-20251001 for speed/cost
# MAX_RETRIES      = 3
# RETRY_DELAY_SEC  = 5

# ──────────────────────────────────────────────
# LOCAL MODEL CALL (OLLAMA)
# ──────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3:8b"

LABEL_COLUMNS = [
    "stereotype",
    "vilification",
    "dehumanization",
    "extreme_language",
    "lack_of_empathy",
    "invalidation",
]

# Map CSV column names → display names used in the prompt / parsing
LABEL_DISPLAY = {
    "stereotype":      "Stereotype",
    "vilification":    "Vilification",
    "dehumanization":  "Dehumanization",
    "extreme_language":"Extreme Language",
    "lack_of_empathy": "Lack of Empathy",
    "invalidation":    "Invalidation",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# PROMPT CONSTRUCTION
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are a strict multi-label classifier.

Each label must be evaluated independently.
Multiple labels may be 1 only if they truly apply.

━━━━━━━━━━━━━━━━━━
LABEL DISTINCTIONS
━━━━━━━━━━━━━━━━━━

1) STEREOTYPE
Assigning neutral social traits (e.g., lazy, greedy, weak, emotional)
that describe behavior or personality.

Do NOT mark Stereotype if the text says the group is evil, corrupt,
dangerous, bad, criminal, or harmful.
Those belong to Vilification.

2) VILIFICATION
Describing a group as morally bad, evil, corrupt, criminal, or dangerous to society.
This is moral condemnation.
It does NOT include:
- comparing to animals (that is Dehumanization)
- calling for violence (that is Extreme Language)

3) DEHUMANIZATION
Comparing people to animals, insects, objects, garbage,
or denying they are human.
If this occurs:
Dehumanization = 1
Stereotype = 0

4) EXTREME LANGUAGE
Calling for violence, removal, destruction,
or using highly aggressive hostile exaggeration.
If there is no call for violence,
do NOT mark Extreme Language.

5) LACK OF EMPATHY
Dismissing, mocking, or minimizing someone’s suffering.
Not moral condemnation.
Not trait assignment.

6) INVALIDATION
Denying someone’s identity or lived experience.
Example: “That discrimination is fake.”

━━━━━━━━━━━━━━━━━━
CONFLICT RULES
━━━━━━━━━━━━━━━━━━

• If text compares a group to animals or objects:
  Dehumanization = 1
  Stereotype = 0

• If text morally condemns a group but does NOT call for violence:
  Vilification = 1
  Extreme Language = 0

• If text calls for violence or removal:
  Extreme Language = 1

• Stereotype only applies to trait generalization,
  not moral evil, not animals, not violence.

━━━━━━━━━━━━━━━━━━

Return exactly six lines:

Stereotype: 0 or 1
Vilification: 0 or 1
Dehumanization: 0 or 1
Extreme Language: 0 or 1
Lack of Empathy: 0 or 1
Invalidation: 0 or 1

No explanation.
"""


def build_user_prompt(text: str) -> str:
    return f"""""
Example 1:
Text: All of them are lazy and useless.
Stereotype: 1
Vilification: 0
Dehumanization: 0
Extreme Language: 0
Lack of Empathy: 0
Invalidation: 0

Example 2:
Text: They are animals and should be removed.
Stereotype: 0
Vilification: 1
Dehumanization: 1
Extreme Language: 1
Lack of Empathy: 0
Invalidation: 0

Example 3:
Text: That discrimination does not exist.
Stereotype: 0
Vilification: 0
Dehumanization: 0
Extreme Language: 0
Lack of Empathy: 0
Invalidation: 1

Now classify:


Text:
\"\"\"{text}\"\"\"

Definitions:
Stereotype = Assigning a fixed social trait (e.g., lazy, greedy, weak, emotional)
  to an entire group.
  This does NOT include comparing them to animals or objects.
Vilification = calling a group evil or dangerous.
Dehumanization = comparing people to animals/objects.
Extreme Language = violent or highly exaggerated hostility.
Lack of Empathy = dismissing or mocking suffering.
Invalidation = denying lived experience or identity.

Classify now."""
# \"\"\"{text}\"\"\"

# Label Definitions:
# - Stereotype:
#   A general statement about ALL members of a group (e.g., "All X are lazy").
# - Vilification:
#   Saying a group is evil, immoral, corrupt, or dangerous.
# - Dehumanization:
#   Comparing people to animals, insects, objects, or denying they are human.
# - Extreme Language:
#   Calling for violence or using extreme hostile exaggeration.
# - Lack of Empathy:
#   Minimizing or mocking someone’s suffering.
# - Invalidation:
#   Denying someone’s identity or lived experience (e.g., "That discrimination is not real").


# Instructions:
# For each label, output either 0 (No) or 1 (Yes).
# Do NOT output "0/1".
# Do NOT output explanations.
# Do NOT output anything except the six lines below.

# First internally evaluate each category separately.
# Then output the final numeric decisions.

# Output format must be exactly:
# Stereotype: <0 or 1>
# Vilification: <0 or 1>
# Dehumanization: <0 or 1>
# Extreme Language: <0 or 1>
# Lack of Empathy: <0 or 1>
# Invalidation: <0 or 1>

# Now classify the given text:

# Stereotype:
# Vilification:
# Dehumanization:
# Extreme Language:
# Lack of Empathy:
# Invalidation:"""

# ──────────────────────────────────────────────
# API CALL
# ──────────────────────────────────────────────
# client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def call_model(text: str) -> str | None:
    try:
        prompt = SYSTEM_PROMPT + "\n\n" + build_user_prompt(text)

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "top_p": 1
                }
            }
        )


        # response = requests.post(
        #     OLLAMA_URL,
        #     json={
        #         "model": MODEL,
        #         "prompt": prompt,
        #         "stream": False
        #     }
        # )

        return response.json()["response"].strip()

    except Exception as e:
        print("Model error:", e)
        return None

# def call_claude(text: str) -> str | None:
#     """Call Claude API with retry logic. Returns raw model response text."""
#     for attempt in range(1, MAX_RETRIES + 1):
#         try:
#             response = client.messages.create(
#                 model=MODEL,
#                 max_tokens=200,
#                 system=SYSTEM_PROMPT,
#                 messages=[{"role": "user", "content": build_user_prompt(text)}],
#             )
#             return response.content[0].text.strip()
#         except anthropic.RateLimitError:
#             print(f"  Rate limit hit. Waiting {RETRY_DELAY_SEC * attempt}s (attempt {attempt}/{MAX_RETRIES})...")
#             time.sleep(RETRY_DELAY_SEC * attempt)
#         except anthropic.APIError as e:
#             print(f"  API error on attempt {attempt}: {e}")
#             time.sleep(RETRY_DELAY_SEC)
#     return None


# ──────────────────────────────────────────────
# RESPONSE PARSING
# ──────────────────────────────────────────────
# Regex patterns for each display label
LABEL_PATTERNS = {col: re.compile(rf"{re.escape(disp)}:\s*([01])", re.IGNORECASE)
                  for col, disp in LABEL_DISPLAY.items()}

def parse_response(response: str) -> dict[str, int | None]:
    """
    Robust prediction parser.
    Handles:
    - Labeled format (Stereotype: 1)
    - Variants (:, -, =)
    - Standalone numeric lines
    - Mixed outputs
    """

    preds = {col: None for col in LABEL_COLUMNS}

    # 1️⃣ Try flexible labeled matching
    for col, disp in LABEL_DISPLAY.items():
        pattern = re.compile(
            rf"{re.escape(disp)}\s*[:=\-]\s*([01])",
            re.IGNORECASE
        )
        match = pattern.search(response)
        if match:
            preds[col] = int(match.group(1))

    # 2️⃣ If some still None → try line-based numeric extraction
    remaining = [col for col, val in preds.items() if val is None]

    if remaining:
        lines = response.strip().split("\n")
        numeric_lines = [line.strip() for line in lines if line.strip() in ["0", "1"]]

        if len(numeric_lines) == 6:
            for i, col in enumerate(LABEL_COLUMNS):
                preds[col] = int(numeric_lines[i])

    # 3️⃣ Final fallback: global number scan
    if any(val is None for val in preds.values()):
        numbers = re.findall(r"\b[01]\b", response)
        if len(numbers) >= 6:
            for i, col in enumerate(LABEL_COLUMNS):
                preds[col] = int(numbers[i])

    return preds


# ──────────────────────────────────────────────
# DATASET LOADING
# ──────────────────────────────────────────────
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Validate required columns
    required = ["translated_te"] + LABEL_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    # Keep only rows with non-null Telugu    
    df = df.dropna(subset=["translated_te"]).reset_index(drop=True)

    # Ensure label columns are integers
    for col in LABEL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    print(f"Loaded {len(df)} rows from '{path}'")
    print(f"Label distribution:\n{df[LABEL_COLUMNS].sum().to_string()}\n")
    return df


# ──────────────────────────────────────────────
# METRICS COMPUTATION
# ──────────────────────────────────────────────
def compute_per_label_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "label":     label,
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "weighted_f1": round(weighted_f1, 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }



# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────
def run_phase1():
    df = load_dataset(DATA_PATH)
    # df = df.head(50)


    pred_cols  = {col: [] for col in LABEL_COLUMNS}
    raw_outputs = []
    parse_failures = 0

    print(f"Running inference on {len(df)} samples using model: {MODEL}\n")
    for idx, row in df.iterrows():
        text = str(row["translated_te"])
        print(f"[{idx+1}/{len(df)}] Processing...", end="\r")

        raw = call_model(text)
        raw_outputs.append(raw if raw else "API_ERROR")

        if raw:
            preds = parse_response(raw)
        else:
            preds = {col: None for col in LABEL_COLUMNS}

        for col in LABEL_COLUMNS:
            val = preds.get(col)
            if val is None:
                parse_failures += 1
                val = 0   # default to 0 on parse failure
            pred_cols[col].append(val)

        # time.sleep(0.3)   # gentle rate limiting

    print(f"\n\nInference complete. Parse failures: {parse_failures}")

    # 🔥 Print predicted positives
    print("\nPredicted positives per label:")
    for col in LABEL_COLUMNS:
        print(f"{col}: {sum(pred_cols[col])}")

    # ── Attach predictions to dataframe
    results_df = df.copy()
    results_df["raw_model_output"] = raw_outputs

    for col in LABEL_COLUMNS:
        results_df[f"pred_{col}"] = pred_cols[col]
    # for col in LABEL_COLUMNS:
    #     results_df[f"pred_{col}"] = pred_cols[col]

    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"Predictions saved to: {RESULTS_CSV}")

    # ── Compute metrics
    metrics_rows = []

    print("Length of df:", len(df))
    print("Length of pred_cols per label:", {k: len(v) for k,v in pred_cols.items()})
    print("Columns in results_df:", results_df.columns.tolist())

    
    # 🔥 DEBUG: Check columns before computing metrics
    print("\nDEBUG — results_df columns:")
    print(results_df.columns.tolist())
    print("\n" + "="*70)
    print("PER-LABEL METRICS")
    print("="*70)
    for col in LABEL_COLUMNS:
        y_true = results_df[col].values
        y_pred = np.array(results_df[f"pred_{col}"].values)
        m = compute_per_label_metrics(y_true, y_pred, LABEL_DISPLAY[col])
        metrics_rows.append(m)
        print(f"\n{m['label']}")
        print(f"  Accuracy : {m['accuracy']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall   : {m['recall']:.4f}")
        print(f"  F1-Score : {m['f1']:.4f}")
        print(f"  Weighted F1: {m['weighted_f1']:.4f}")
        print(f"  TP={m['TP']}  FP={m['FP']}  TN={m['TN']}  FN={m['FN']}")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)
    print(f"\nMetrics saved to: {METRICS_CSV}")

    # ── Macro averages
    print("\n" + "="*70)
    print("MACRO AVERAGES")
    print("="*70)
    for metric in ["accuracy", "precision", "recall", "f1", "weighted_f1"]:
        print(f"  Macro {metric:10s}: {metrics_df[metric].mean():.4f}")


    return results_df, metrics_df

# ──────────────────────────────────────────────
# TEST SINGLE SENTENCE
# ──────────────────────────────────────────────
def test_single_text():
    # test_text="ఇవాళ వాళ్లు అందరూ చాలా సోమరులు మరియు పనికిరాని వారు." #stereotype
    # test_text = "వాళ్లు మనుషులు కాదు, వారు జంతువుల్లా ప్రవర్తిస్తున్నారు." #dehumanization
    test_text = "వాళ్లు చాలా చెడ్డవాళ్లు మరియు సమాజానికి ప్రమాదకరులు."

    print("\n" + "="*70)
    print("TESTING SINGLE SENTENCE")
    print("="*70)
    print("Input Text:")
    print(test_text)
    print("\nCalling model...\n")

    raw = call_model(test_text)

    if raw:
        print("Raw Model Output:\n")
        print(raw)
        print("\nParsed Prediction:\n")
        parsed = parse_response(raw)
        for label, value in parsed.items():
            print(f"{LABEL_DISPLAY[label]}: {value}")
    else:
        print("Model call failed.")



if __name__ == "__main__":
    # run_phase1()
    test_single_text()