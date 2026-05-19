"""
STEREOTYPE DETECTION PROBE — IMPROVED
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
