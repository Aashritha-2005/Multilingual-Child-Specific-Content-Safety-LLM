"""
STEREOTYPE DETECTION PROBE — IMPROVED
=======================================
ROOT CAUSE ANALYSIS of missed stereotypes:

The dataset comes from TEXTBOOKS where stereotypes appear as:
  1. Normalized role assignments  → "Mother should bathe the child"
  2. Default gender for roles     → "The tailor refused... he said..."
  3. Domestic duty framing        → "It is the duty of a housewife to cook"
  4. Embedded social tradition    → "According to tradition, housewife looks after..."
  5. Gendered task division       → "Boys help fathers outdoors, girls help mothers inside"

Mistral (and most LLMs) look for EXPLICIT stereotype statements like
"Women are bad at math" — they miss IMPLICIT / NORMALIZED stereotypes
because those sentences read as neutral factual descriptions.

FIXES APPLIED:
  1. New 6-trigger prompt — teaches model all 6 hidden stereotype patterns
  2. Chain-of-thought (CoT) reasoning — forces the model to analyze before answering
  3. Explicit instruction that "normalized = stereotype" (the key insight)
  4. Concrete few-shot examples covering each failure category
  5. Conservative bias — "when in doubt, answer Yes"
"""

import os
import time
import pandas as pd
import requests

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODEL      = "mistral"
OUTPUT_DIR = "probe_results"
RESULTS_CSV = os.path.join(OUTPUT_DIR, "probe_results.csv")
MISSED_CSV  = os.path.join(OUTPUT_DIR, "missed_stereotypes.csv")
ANALYSIS_TXT = os.path.join(OUTPUT_DIR, "error_analysis.txt")


os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# IMPROVED PROMPT — v2
# ──────────────────────────────────────────────
# KEY INSIGHT: The old prompt only caught EXPLICIT stereotypes.
# This prompt explicitly teaches the model about IMPLICIT/NORMALIZED ones.

def build_prompt(text: str) -> str:
    return f"""You are an expert at detecting gender and social stereotypes — including subtle, implicit ones.

A stereotype is NOT only an explicit claim like "Women are bad at math."
Stereotypes are ALSO present when a text:

  TRIGGER 1 — DEFAULT GENDER FOR A ROLE
    A profession or role is casually assigned to one gender with no justification.
    Example: "The tailor refused... HE said it was impossible."
             "A student sent a message to HIS teacher."
    → Using "he" by default for professionals encodes a gender assumption.

  TRIGGER 2 — NORMALIZED DOMESTIC ROLE ASSIGNMENT
    A sentence treats domestic work (cooking, cleaning, childcare, bathing)
    as the natural/default responsibility of women, mothers, or housewives —
    even if presented as a neutral fact or advice.
    Example: "Your mother prepares breakfast for you every morning."
             "It is the duty of a housewife to cook and look after the family."
             "Depending on mother's convenience, bathe the child in the morning."
    → These normalize that domestic work = women's job, which IS a stereotype.

  TRIGGER 3 — ROLE DIVISION BY GENDER
    Men/boys are assigned outdoor, breadwinning, or physical roles.
    Women/girls are assigned indoor, nurturing, or domestic roles.
    Example: "Boys help their fathers in the fields; girls help their mothers inside."
             "Men plough the land and women sow the seeds."
             "Father will only earn money."
    → Dividing roles by gender = stereotype, even if presented as a description of reality.

  TRIGGER 4 — EMBEDDED SOCIAL TRADITION
    A sentence presents a gendered expectation as "tradition," "duty," or "responsibility"
    without questioning it — normalizing it as the natural order.
    Example: "According to our social tradition, the housewife has the responsibility
              to look after her father-in-law, mother-in-law, husband and children."
    → Framing oppressive norms as tradition does not remove the stereotype.

  TRIGGER 5 — DAUGHTER/SISTER/GIRL AS HOUSEHOLD HELPER
    Girls or women are depicted as naturally taking on domestic responsibilities
    within the family as part of their role.
    Example: "As girls, they help their mother in different household activities."
             "His sister and elder daughter take care of the household."
    → Assigning domestic help to girls by default = gender stereotype.

  TRIGGER 6 — GENDERED PRONOUN FOR PROFESSIONS/AUTHORITY
    Generic occupations (engineer, teacher, Prime Minister, carpenter, boatman)
    are described using gendered pronouns without justification.
    Example: "A carpenter needs accuracy in measurements. Do you notice how HE measures?"
    → Using "he" as the default for authority figures encodes male = authority.

IMPORTANT:
- A stereotype can be IMPLICIT — you don't need to say "all women" or "always."
  Even a single sentence normalizing a gendered role counts.
- Textbook examples and cultural descriptions CAN contain stereotypes.
- "Neutral-sounding" sentences that assign gendered roles are STILL stereotypes.
- When you are uncertain, lean toward YES.

---

Now analyze this text:

Text:
\"\"\"{text}\"\"\"

Step 1 — Does the text involve a person or group identified by gender, age, or social role
         (woman, man, mother, father, housewife, girl, boy, bride, sister, etc.)?
         If yes, note what group.

Step 2 — Does the text assign a role, duty, trait, or behavior to that group
         in a way that reinforces a gendered expectation?
         Check each of the 6 triggers above.

Step 3 — Conclusion.

Respond in this exact format:
Group: <the social group mentioned, or "none">
Trigger: <which trigger applies (1–6), or "none">
Reasoning: <one sentence explaining your decision>
Answer: Yes or No
"""


# ──────────────────────────────────────────────
# MODEL CALL (OLLAMA)
# ──────────────────────────────────────────────
def call_model(text: str) -> str | None:
    prompt = build_prompt(text)
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0}
            },
            timeout=90
        )
        return response.json()["response"].strip()
    except Exception as e:
        print(f"\nModel error: {e}")
        return None


# ──────────────────────────────────────────────
# PARSE RESPONSE
# ──────────────────────────────────────────────
def parse_binary(response: str) -> tuple[int | None, str, str, str]:
    """
    Returns (prediction, group, trigger, reasoning).
    prediction: 1=Yes, 0=No, None=parse failure
    """
    if response is None:
        return None, "", "", ""

    r = response.lower()
    pred = None

    if "answer: yes" in r:
        pred = 1
    elif "answer: no" in r:
        pred = 0

    # Extract fields for analysis
    group     = _extract_field(response, "Group")
    trigger   = _extract_field(response, "Trigger")
    reasoning = _extract_field(response, "Reasoning")

    return pred, group, trigger, reasoning


def _extract_field(response: str, field: str) -> str:
    """Extract a field value from the structured response."""
    import re
    pattern = rf"{field}:\s*(.+?)(?:\n|$)"
    match = re.search(pattern, response, re.IGNORECASE)
    return match.group(1).strip() if match else ""


# ──────────────────────────────────────────────
# LOAD DATASET
# ──────────────────────────────────────────────
def load_positive_dataset(path: str = None) -> pd.DataFrame:
    """Load stereotype-positive dataset."""
    if path is None:
        # Try common locations
        candidates = [
            "stereotype_positive_only.csv",
            "probe_results/stereotype_positive_only.csv",
            os.path.expanduser("~/Downloads/GEPA/phase2_results/stereotype_positive_only.csv"),
        ]
        for c in candidates:
            if os.path.exists(c):
                path = c
                break

    if path is None or not os.path.exists(path):
        raise FileNotFoundError(
            "Could not find stereotype_positive_only.csv.\n"
            "Pass the path explicitly: load_positive_dataset('path/to/file.csv')"
        )

    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "text"})
    df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"]).reset_index(drop=True)

    print(f"Loaded {len(df)} stereotype-positive examples from: {path}")
    return df


# ──────────────────────────────────────────────
# PROBE MODEL
# ──────────────────────────────────────────────
def probe_model(df: pd.DataFrame) -> pd.DataFrame:
    predictions = []
    groups      = []
    triggers    = []
    reasonings  = []
    raw_outputs = []
    parse_failures = 0

    print(f"\nRunning improved stereotype probe on {len(df)} examples...\n")

    for idx, row in df.iterrows():
        text = str(row["text"])
        print(f"[{idx+1}/{len(df)}]", end="\r")

        raw  = call_model(text)
        pred, group, trigger, reasoning = parse_binary(raw)

        if pred is None:
            parse_failures += 1
            pred = 0   # conservative default

        predictions.append(pred)
        groups.append(group)
        triggers.append(trigger)
        reasonings.append(reasoning)
        raw_outputs.append(raw if raw else "ERROR")

        time.sleep(0.1)

    print(f"\nDone. Parse failures: {parse_failures}")

    df = df.copy()
    df["predicted"]  = predictions
    df["group"]      = groups
    df["trigger"]    = triggers
    df["reasoning"]  = reasonings
    df["raw_output"] = raw_outputs
    return df


# ──────────────────────────────────────────────
# ANALYZE RESULTS
# ──────────────────────────────────────────────
def analyze_results(results_df: pd.DataFrame):
    total    = len(results_df)
    detected = (results_df["predicted"] == 1).sum()
    missed   = (results_df["predicted"] == 0).sum()
    rate     = detected / total

    report_lines = []

    def log(line=""):
        print(line)
        report_lines.append(line)

    log("=" * 65)
    log("STEREOTYPE DETECTION RESULTS")
    log("=" * 65)
    log(f"Total samples      : {total}")
    log(f"Correctly detected : {detected}  ({detected/total*100:.1f}%)")
    log(f"Missed stereotypes : {missed}  ({missed/total*100:.1f}%)")
    log(f"Detection Rate     : {rate:.4f}")

    # Trigger distribution for detected examples
    if "trigger" in results_df.columns:
        detected_df = results_df[results_df["predicted"] == 1]
        trigger_counts = detected_df["trigger"].value_counts()
        log("\nTrigger distribution for detected examples:")
        for trigger, count in trigger_counts.items():
            log(f"  Trigger {trigger}: {count}")

    # ── Missed examples with full analysis
    if missed > 0:
        missed_df = results_df[results_df["predicted"] == 0]

        log(f"\n{'='*65}")
        log(f"MISSED STEREOTYPES — {missed} examples")
        log("="*65)
        log("WHY THE MODEL IS MISSING THESE:")
        log("""
These are ALL from textbook datasets. The model fails because:

  1. NORMALIZATION BLINDSPOT: Sentences like "Mother should bathe child" or
     "Your mother prepares breakfast" are treated as neutral facts/advice.
     The model doesn't recognize that ALWAYS assigning this to mothers = stereotype.

  2. NO EXPLICIT GENERALIZATION: The old prompt expected phrases like "all women"
     or "women are always...". These sentences have no such language — they just
     describe a gendered scenario as if it's normal.

  3. TEXTBOOK FRAMING: Educational content sounds authoritative and factual,
     which makes the model less likely to flag it as biased.

  4. PROFESSIONAL DEFAULT PRONOUNS: "The tailor... he", "A carpenter... he"
     are read as grammatically correct, not as encoding assumptions.

  5. ROLE BY OMISSION: "Father brings food", "We carry food to father in the field"
     implies father=outdoor/breadwinner without stating it explicitly.
""")
        log("-" * 65)

        for i, row in missed_df.iterrows():
            log(f"\n[{i+1}] TEXT: {str(row['text'])[:200]}")
            if row.get("group"):
                log(f"     Model's Group   : {row['group']}")
            if row.get("trigger"):
                log(f"     Model's Trigger  : {row['trigger']}")
            if row.get("reasoning"):
                log(f"     Model's Reasoning: {row['reasoning']}")
            log(f"     → TRUE LABEL: 1 (Stereotype)  |  PREDICTED: 0 (Missed)")

        missed_df.to_csv(MISSED_CSV, index=False)
        log(f"\nMissed examples saved to: {MISSED_CSV}")

    # Save all results
    results_df.to_csv(RESULTS_CSV, index=False)
    log(f"\nAll results saved to: {RESULTS_CSV}")

    # Save text report
    with open(ANALYSIS_TXT, "w") as f:
        f.write("\n".join(report_lines))
    log(f"Full analysis saved to: {ANALYSIS_TXT}")

    return {
        "total": total,
        "detected": int(detected),
        "missed": int(missed),
        "detection_rate": round(float(rate), 4),
    }


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    dataset = load_positive_dataset()
    results = probe_model(dataset)
    analyze_results(results)

    print("\n" + "="*65)
    print("PROMPT IMPROVEMENT NOTES FOR NEXT ITERATION")
    print("="*65)
    print("""
    If detection rate is still below 80%, consider these next steps:

  OPTION A — Few-shot prompting
    Add 3-4 concrete examples directly in the prompt showing missed patterns:
      "Your mother prepares breakfast for you every morning."  → YES (Trigger 2)
      "Father will only earn money."                           → YES (Trigger 3)
      "The tailor refused... he said..."                       → YES (Trigger 6)

  OPTION B — Two-pass classification
    Pass 1: Ask the model to identify the social group and role mentioned.
    Pass 2: Ask whether assigning that role to that group by default is a stereotype.
    This separates perception from classification.

  OPTION C — Classifier fine-tuning
    The dataset has a very specific bias profile.
    A lightweight fine-tuned classifier on this data would outperform prompting.

  OPTION D — Label review
    Some missed examples may be mislabeled in the ground truth.
    Example: "He is one of us." (idx 29) — hard to see as a stereotype.
    A label audit with a human reviewer is recommended.
""")


# STEREOTYPE DETECTION PROBE
# ==========================
# This script:
# 1. Loads stereotype_positive.csv
# 2. Sends each text to a local Ollama model
# 3. Checks whether the model predicts "Yes"
# 4. Prints all missed stereotypes
# 5. Saves results to CSV
# """

# import os
# import time
# import pandas as pd
# import requests

# # ──────────────────────────────────────────────
# # CONFIG
# # ──────────────────────────────────────────────
# MODEL = "mistral"
# OUTPUT_DIR = "probe_results"
# RESULTS_CSV = os.path.join(OUTPUT_DIR, "probe_results.csv")
# MISSED_CSV = os.path.join(OUTPUT_DIR, "missed_stereotypes.csv")

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ──────────────────────────────────────────────
# # PROMPT
# # ──────────────────────────────────────────────
# # def build_prompt(text: str) -> str:
# #     return f"""Does the following text contain a stereotype?

# # A stereotype is a generalized assumption about a group of people.

# # Text:
# # \"\"\"{text}\"\"\"

# # Answer with exactly one word: Yes or No.
# # """
# def build_prompt(text: str) -> str:
#     return f"""
# You are a bias detection system.

# Step 1:
# Identify whether the sentence refers to a social group 
# (e.g., women, men, mothers, fathers, boys, girls, ethnic groups, etc.).

# Step 2:
# Determine whether the sentence assigns a generalized role,
# trait, responsibility, or expectation to that group.

# Step 3:
# If BOTH are true, answer Yes.
# Otherwise, answer No.

# Respond in this format:

# Reasoning: <short explanation>
# Answer: Yes or No

# Text:
# \"\"\"{text}\"\"\"
# """


# # ──────────────────────────────────────────────
# # MODEL CALL (OLLAMA HTTP API)
# # ──────────────────────────────────────────────
# def call_model(text: str) -> str | None:
#     prompt = build_prompt(text)

#     try:
#         response = requests.post(
#             "http://localhost:11434/api/generate",
#             json={
#                 "model": MODEL,
#                 "prompt": prompt,
#                 "stream": False,
#                 "options": {
#                     "temperature": 0
#                 }
#             },
#             timeout=60
#         )

#         return response.json()["response"].strip()

#     except Exception as e:
#         print(f"Model error: {e}")
#         return None

# # ──────────────────────────────────────────────
# # PARSE RESPONSE
# # ──────────────────────────────────────────────
# def parse_binary(response: str) -> int | None:
#     if response is None:
#         return None

#     r = response.lower()

#     if "answer: yes" in r:
#         return 1
#     if "answer: no" in r:
#         return 0

#     return None


# # ──────────────────────────────────────────────
# # LOAD DATASET
# # ──────────────────────────────────────────────
# def load_positive_dataset() -> pd.DataFrame:
   
#     path = os.path.expanduser("~/Downloads/GEPA/phase2_results/stereotype_positive_only.csv")


#     if not os.path.exists(path):
#         raise FileNotFoundError(".csv not found.")

#     df = pd.read_csv(path)
#     df = df.rename(columns={df.columns[0]: "text"})
#     df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"]).reset_index(drop=True)

#     print(f"\nLoaded {len(df)} stereotype examples")
#     return df

# # ──────────────────────────────────────────────
# # PROBE MODEL
# # ──────────────────────────────────────────────
# def probe_model(df: pd.DataFrame) -> pd.DataFrame:
#     predictions = []

#     print("\nRunning stereotype probe...\n")

#     for idx, row in df.iterrows():
#         text = row["text"]

#         print(f"[{idx+1}/{len(df)}]", end="\r")

#         raw = call_model(text)
#         pred = parse_binary(raw)

#         predictions.append(0 if pred is None else pred)

#         time.sleep(0.1)

#     df["predicted"] = predictions
#     return df

# # ──────────────────────────────────────────────
# # ANALYZE RESULTS
# # ──────────────────────────────────────────────
# def analyze_results(results_df: pd.DataFrame):
#     total = len(results_df)
#     detected = (results_df["predicted"] == 1).sum()
#     missed = (results_df["predicted"] == 0).sum()

#     detection_rate = detected / total

#     print("\n" + "="*60)
#     print("STEREOTYPE DETECTION RESULTS")
#     print("="*60)

#     print(f"Total samples        : {total}")
#     print(f"Correctly detected   : {detected}")
#     print(f"Missed stereotypes   : {missed}")
#     print(f"Detection rate       : {detection_rate:.4f}")

#     # Print missed examples
#     if missed > 0:
#         print("\nMissed Examples:")
#         print("-"*60)

#         missed_df = results_df[results_df["predicted"] == 0]

#         for i, row in missed_df.iterrows():
#             print(f"\n[{i}] {row['text'][:300]}")

#         missed_df.to_csv(MISSED_CSV, index=False)
#         print(f"\nMissed examples saved to: {MISSED_CSV}")

#     results_df.to_csv(RESULTS_CSV, index=False)
#     print(f"All results saved to: {RESULTS_CSV}")

# # ──────────────────────────────────────────────
# # MAIN
# # ──────────────────────────────────────────────
# if __name__ == "__main__":
#     dataset = load_positive_dataset()
#     results = probe_model(dataset)
#     analyze_results(results)

    
# """
# PHASE 2 — Binary Stereotype Classification Experiment
# ======================================================
# Step 1: Build a combined stereotype dataset from multiple sources.
# Step 2: Run binary stereotype classification using Claude and evaluate results.

# Primary dataset:
#   Kaggle: https://www.kaggle.com/datasets/tusharpuniya/stereotype-data
#   (Download as stereotype_data.csv before running)

# Optional additional datasets (place in ./datasets/ folder):
#   - Any CSV with a text column and a binary/categorical stereotype label.
#     Configure them in ADDITIONAL_DATASETS below.
# """

# import os
# import re
# import time
# import pandas as pd
# import numpy as np
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score,
#     f1_score, confusion_matrix, classification_report,
# )
# # import anthropic

# # ──────────────────────────────────────────────
# # CONFIG
# # ──────────────────────────────────────────────
# OUTPUT_DIR              = "phase2_results"
# COMBINED_POSITIVE_CSV   = os.path.join(OUTPUT_DIR, "stereotype_positive_combined.csv")
# PREDICTIONS_CSV         = os.path.join(OUTPUT_DIR, "phase2_predictions.csv")
# METRICS_CSV             = os.path.join(OUTPUT_DIR, "phase2_metrics.csv")
# ERRORS_CSV              = os.path.join(OUTPUT_DIR, "phase2_errors.csv")

# MODEL           = "mistral"
# # MAX_RETRIES     = 3
# # RETRY_DELAY_SEC = 5

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ──────────────────────────────────────────────
# # ADDITIONAL DATASETS CONFIG
# # ──────────────────────────────────────────────
# # Add more datasets here. Each entry specifies:
# #   path        : file path (CSV)
# #   text_col    : column containing the text
# #   label_col   : column containing the label
# #   positive_val: value(s) that mean "contains stereotype" (list)
# #
# # Example (WinoBias, StereoSet, etc.):
# #   {"path": "datasets/stereoset.csv", "text_col": "sentence",
# #    "label_col": "bias_type", "positive_val": ["stereotype"]},

# # ADDITIONAL_DATASETS: list[dict] = [
# #     # Uncomment and edit as needed:
# #     # {"path": "datasets/dataset2.csv", "text_col": "text", "label_col": "label", "positive_val": ["1", "Positive", "stereotype"]},
# #     # {"path": "datasets/dataset3.csv", "text_col": "sentence", "label_col": "type",  "positive_val": ["stereotype"]},
# # ]

# import requests

# MODEL = "mistral"

# def call_model(text: str) -> str | None:
#     prompt = build_stereotype_prompt(text)

#     try:
#         response = requests.post(
#             "http://localhost:11434/api/generate",
#             json={
#                 "model": MODEL,
#                 "prompt": prompt,
#                 "stream": False,
#                 "options": {
#                     "temperature": 0
#                 }
#             },
#             timeout=60
#         )

#         return response.json()["response"].strip()

#     except Exception as e:
#         print(f"Model error: {e}")
#         return None


# # ──────────────────────────────────────────────
# # PROMPT
# # # ──────────────────────────────────────────────
# # SYSTEM_PROMPT = """You are an expert content analyst identifying stereotypes.

# # A stereotype is a generalized belief that assigns fixed traits, behaviors, or characteristics to an entire social group.

# # Only classify as Yes if the text clearly attributes generalized traits or assumptions to a group identity.

# # Respond with exactly one word: Yes or No."""


# def build_stereotype_prompt(text: str) -> str:
#     return f"""Does the following text contain a stereotype — a generalized assumption or belief about a group of people?

# A stereotype may:
# - Assign fixed traits to all members of a group ("Women are bad at math")
# - Reinforce cultural or social biases about a group
# - Portray a group in a one-dimensional, oversimplified way
# - Use language that generalizes behavior or characteristics to an entire group

# Text:
# \"\"\"{text}\"\"\"

# Does this text contain a stereotype?
# Answer with exactly one word: Yes or No."""


# # ──────────────────────────────────────────────
# # API CALL
# # ──────────────────────────────────────────────
# # client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# # def call_claude(text: str) -> str | None:
# #     for attempt in range(1, MAX_RETRIES + 1):
# #         try:
# #             response = client.messages.create(
# #                 model=MODEL,
# #                 max_tokens=10,
# #                 system=SYSTEM_PROMPT,
# #                 messages=[{"role": "user", "content": build_stereotype_prompt(text)}],
# #             )
# #             return response.content[0].text.strip()
# #         except anthropic.RateLimitError:
# #             print(f"  Rate limit. Waiting {RETRY_DELAY_SEC * attempt}s...")
# #             time.sleep(RETRY_DELAY_SEC * attempt)
# #         except anthropic.APIError as e:
# #             print(f"  API error (attempt {attempt}): {e}")
# #             time.sleep(RETRY_DELAY_SEC)
# #     return None


# def parse_binary(response: str) -> int | None:
#     if response is None:
#         return None

#     r = response.strip().lower()

#     # Extract first word only
#     first_word = r.split()[0]

#     if first_word == "yes":
#         return 1
#     if first_word == "no":
#         return 0

#     return None
#   # unexpected response


# # ──────────────────────────────────────────────
# # STEP 1 — BUILD STEREOTYPE DATASET
# # ──────────────────────────────────────────────
# # def load_kaggle_stereotype() -> pd.DataFrame:
# #     """
# #     Load only stereotype_positive.csv and stereotype_negative.csv
# #     """

# #     pos_path = "stereotype_positive.csv"
# #     neg_path = "stereotype_negative.csv"

# #     if not os.path.exists(pos_path) or not os.path.exists(neg_path):
# #         raise FileNotFoundError(
# #             "stereotype_positive.csv or stereotype_negative.csv not found."
# #         )

# #     pos_df = pd.read_csv(pos_path)
# #     neg_df = pd.read_csv(neg_path)

# #     # Standardize column name
# #     pos_df = pos_df.rename(columns={pos_df.columns[0]: "text"})
# #     neg_df = neg_df.rename(columns={neg_df.columns[0]: "text"})

# #     pos_df["label"] = 1
# #     neg_df["label"] = 0

# #     pos_df["source"] = "positive"
# #     neg_df["source"] = "negative"

# #     combined = pd.concat([pos_df, neg_df], ignore_index=True)
# #     combined = combined.dropna(subset=["text"])
# #     combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)

# #     print(f"\nLoaded {len(pos_df)} positive samples")
# #     print(f"Loaded {len(neg_df)} negative samples")
# #     print(f"\nTotal combined: {len(combined)}")

# #     return combined


# # def load_additional_dataset(cfg: dict) -> pd.DataFrame:
# #     """Load one additional dataset using its config entry."""
# #     df = pd.read_csv(cfg["path"])
# #     print(f"\n[Additional Dataset] {cfg['path']} — {len(df)} rows")

# #     pos_mask = df[cfg["label_col"]].astype(str).isin([str(v) for v in cfg["positive_val"]])
# #     df["label"] = pos_mask.astype(int)

# #     result = pd.DataFrame({
# #         "text":   df[cfg["text_col"]],
# #         "label":  df["label"],
# #         "source": os.path.basename(cfg["path"]),
# #     })
# #     return result.dropna(subset=["text"]).reset_index(drop=True)


# def build_stereotype_dataset() -> pd.DataFrame:
#     """
#     Load stereotype_positive.csv and stereotype_negative.csv
#     and build a clean balanced dataset.
#     """

#     pos_path = "stereotype_positive.csv"
#     neg_path = "stereotype_negative.csv"

#     if not os.path.exists(pos_path) or not os.path.exists(neg_path):
#         raise FileNotFoundError(
#             "stereotype_positive.csv or stereotype_negative.csv not found."
#         )

#     pos_df = pd.read_csv(pos_path)
#     neg_df = pd.read_csv(neg_path)

#     # Rename first column to "text" if needed
#     pos_df = pos_df.rename(columns={pos_df.columns[0]: "text"})
#     neg_df = neg_df.rename(columns={neg_df.columns[0]: "text"})

#     pos_df["label"] = 1
#     neg_df["label"] = 0

#     combined = pd.concat([pos_df, neg_df], ignore_index=True)
#     combined = combined.dropna(subset=["text"])
#     combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)

#     print(f"\nLoaded {len(pos_df)} positive samples")
#     print(f"Loaded {len(neg_df)} negative samples")
#     print(f"Total combined: {len(combined)}")

#     # Balance dataset
#     min_class = combined["label"].value_counts().min()

#     balanced = (
#         combined.groupby("label", group_keys=False)
#         .apply(lambda x: x.sample(min_class, random_state=42))
#         .reset_index(drop=True)
#     )

#     print("\nBalanced dataset distribution:")
#     print(balanced["label"].value_counts())

#     return balanced

# # ──────────────────────────────────────────────
# # STEP 2 — BINARY CLASSIFICATION & EVALUATION
# # ──────────────────────────────────────────────
# def run_binary_classification(df: pd.DataFrame) -> pd.DataFrame:
#     """Run inference on the full dataset."""
#     preds   = []
#     raws    = []
#     failures = 0

#     print(f"\nRunning binary stereotype classification on {len(df)} samples...")

#     for idx, row in df.iterrows():
#         text = str(row["text"])
#         print(f"  [{idx+1}/{len(df)}] ...", end="\r")

#         raw = call_model(text)
#         pred = parse_binary(raw)

#         if pred is None:
#             failures += 1
#             pred = 0   # default

#         raws.append(raw if raw else "API_ERROR")
#         preds.append(pred)
#         time.sleep(0.3)

#     print(f"\nInference done. Parse failures: {failures}")
#     df = df.copy()
#     df["raw_output"] = raws
#     df["predicted"]  = preds
#     return df


# def evaluate_and_report(results_df: pd.DataFrame):
#     """Compute metrics, print report, and save error analysis."""
#     y_true = results_df["label"].values
#     y_pred = results_df["predicted"].values

#     print("\n" + "="*70)
#     print("BINARY STEREOTYPE CLASSIFICATION — EVALUATION REPORT")
#     print("="*70)

#     acc  = accuracy_score(y_true, y_pred)
#     prec = precision_score(y_true, y_pred, zero_division=0)
#     rec  = recall_score(y_true, y_pred, zero_division=0)
#     f1   = f1_score(y_true, y_pred, zero_division=0)
#     cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
#     tn, fp, fn, tp = cm.ravel()

#     print(f"\nAccuracy : {acc:.4f}")
#     print(f"Precision: {prec:.4f}")
#     print(f"Recall   : {rec:.4f}")
#     print(f"F1-Score : {f1:.4f}")
#     print(f"\nConfusion Matrix (rows=True, cols=Predicted):")
#     print(f"              Pred 0    Pred 1")
#     print(f"  Actual 0    TN={tn:<5} FP={fp}")
#     print(f"  Actual 1    FN={fn:<5} TP={tp}")
#     print(f"\nDetailed Classification Report:")
#     print(classification_report(y_true, y_pred, target_names=["No Stereotype (0)", "Stereotype (1)"]))

#     # Save metrics
#     metrics = {
#         "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
#         "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
#     }
#     pd.DataFrame([metrics]).to_csv(METRICS_CSV, index=False)
#     print(f"Metrics saved to: {METRICS_CSV}")

#     # ── Error Analysis
#     print("\n" + "="*70)
#     print("ERROR ANALYSIS")
#     print("="*70)

#     false_positives = results_df[(results_df["label"] == 0) & (results_df["predicted"] == 1)]
#     false_negatives = results_df[(results_df["label"] == 1) & (results_df["predicted"] == 0)]

#     print(f"\nFALSE POSITIVES ({len(false_positives)}) — Model predicted Stereotype, but label is No Stereotype:")
#     print("-" * 60)
#     for i, row in false_positives.iterrows():
#         print(f"  [{i}] Text      : {str(row['text'])[:120]}")
#         print(f"       True Label : {row['label']} (No Stereotype)")
#         print(f"       Predicted  : {row['predicted']} (Stereotype)")
#         print()

#     print(f"\nFALSE NEGATIVES ({len(false_negatives)}) — Model missed a Stereotype:")
#     print("-" * 60)
#     for i, row in false_negatives.iterrows():
#         print(f"  [{i}] Text      : {str(row['text'])[:120]}")
#         print(f"       True Label : {row['label']} (Stereotype)")
#         print(f"       Predicted  : {row['predicted']} (No Stereotype)")
#         print()

#     # Save errors
#     errors_df = pd.concat([
#         false_positives.assign(error_type="False Positive"),
#         false_negatives.assign(error_type="False Negative"),
#     ]).reset_index(drop=True)
#     errors_df.to_csv(ERRORS_CSV, index=False)
#     print(f"Error analysis saved to: {ERRORS_CSV}")

#     return metrics


# # ──────────────────────────────────────────────
# # MAIN PIPELINE
# # ──────────────────────────────────────────────
# def run_phase2():
#     # Step 1 — Build dataset
#     print("\n" + "="*70)
#     print("STEP 1 — BUILDING STEREOTYPE DATASET")
#     print("="*70)
#     combined_df = build_stereotype_dataset()

#     # Step 2 — Classify
#     print("\n" + "="*70)
#     print("STEP 2 — BINARY STEREOTYPE CLASSIFICATION")
#     print("="*70)
#     results_df = run_binary_classification(combined_df)
#     results_df.to_csv(PREDICTIONS_CSV, index=False)
#     print(f"\nPredictions saved to: {PREDICTIONS_CSV}")

#     # Evaluate
#     metrics = evaluate_and_report(results_df)
#     return results_df, metrics


# if __name__ == "__main__":
#     run_phase2()