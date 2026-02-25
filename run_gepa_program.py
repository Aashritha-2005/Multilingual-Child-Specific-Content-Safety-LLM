"""
run_gepa_program.py
===================
Run GEPA optimization for DSPy program-based harmful speech classification.

This script:
1. Loads Telugu harmful speech dataset
2. Sets up DSPy LM
3. Loads seed program
4. Runs GEPA optimization
5. Saves best evolved program

SETUP:
  export OPENROUTER_API_KEY='...'
  python run_gepa_program.py
"""

import os
import sys
import json
import dspy
import pandas as pd
from pathlib import Path
from gepa import optimize

# Import our adapter
from telugu_program_adapter import TeluguHarmfulSpeechAdapter
from reflection_lm import reflection_lm 

import litellm

def reflection_lm(prompt: str) -> str:
    system_prompt = (
        "You are a Python code generator.\n"
        "STRICT RULES:\n"
        "- Output ONLY valid Python code\n"
        "- Do NOT include explanations\n"
        "- Do NOT include markdown\n"
        "- The FIRST LINE MUST be: import dspy\n"
        "- The output MUST be executable via exec()\n"
    )

    response = litellm.completion(
        model=REFLECTION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=512,
    )

    text = response["choices"][0]["message"]["content"]

    if not text.lstrip().startswith("import dspy"):
        raise RuntimeError("Reflection LM returned invalid Python")

    return text



# ================================================================
# CONFIG
# ================================================================

LANGUAGE = "tel"
CSV_PATH = Path("data/final_telugu_multilabel.csv")
SAMPLE_SIZE = 40
VAL_SIZE = 15
RANDOM_SEED = 42

# Models
TASK_MODEL = "openrouter/google/gemma-3-27b-it"
REFLECTION_MODEL = "openrouter/google/gemma-3-27b-it"  # Or use larger model

# GEPA settings
MAX_METRIC_CALLS = 200  # Increase for better results
OUTPUT_DIR = Path("gepa_program_outputs")

LABELS = [
    "stereotype", "vilification", "dehumanization",
    "extreme_language", "lack_of_empathy", "invalidation"
]


# ================================================================
# SETUP
# ================================================================

def check_api_key():
    """Verify OpenRouter API key is set."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        sys.exit(
            "\n✗ OPENROUTER_API_KEY not set.\n"
            "\n"
            "Get your API key:\n"
            "  1. Visit https://openrouter.ai/\n"
            "  2. Sign up / log in\n"
            "  3. Get API key from dashboard\n"
            "\n"
            "Then run:\n"
            "  export OPENROUTER_API_KEY='sk-or-v1-...'\n"
        )
    print("✓ API key verified")
    return api_key


def load_dataset(csv_path: Path, sample_size: int, val_size: int):
    """
    Load dataset and convert to dspy.Example format.
    
    Returns:
        (trainset, valset) as lists of dspy.Example
    """
    if not csv_path.exists():
        sys.exit(f"\n✗ Dataset not found: {csv_path}\n")
    
    df = pd.read_csv(csv_path)
    
    # Shuffle
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Convert to dspy.Example
    examples = []
    for _, row in df.iterrows():
        example = dspy.Example(
            text=row["translated_te"],  # Use translated Telugu text
            stereotype=int(row['stereotype']),
            vilification=int(row['vilification']),
            dehumanization=int(row['dehumanization']),
            extreme_language=int(row['extreme_language']),
            lack_of_empathy=int(row['lack_of_empathy']),
            invalidation=int(row['invalidation'])
        ).with_inputs('text')  # Only 'text' is input, rest are labels
        print("DEBUG Example fields:", example.__dict__)

        
        examples.append(example)
    
    # Split
    trainset = examples[:sample_size]
    valset = examples[sample_size:sample_size + val_size]
    
    return trainset, valset


def load_seed_program(seed_path: Path = Path("seed_program.py")):
    """Load seed program source code."""
    if not seed_path.exists():
        sys.exit(f"\n✗ Seed program not found: {seed_path}\n")
    
    return seed_path.read_text(encoding="utf-8")


def save_results(result, output_dir: Path):
    """Save optimization results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best program
    best_program = result.best_candidate["program"]
    (output_dir / "best_program.py").write_text(best_program, encoding="utf-8")
    
    # Save summary
    summary = {
        "val_aggregate_scores": result.val_aggregate_scores,
        "best_score": max(result.val_aggregate_scores) if result.val_aggregate_scores else 0.0,
        "task_model": TASK_MODEL,
        "reflection_model": REFLECTION_MODEL,
        "max_metric_calls": MAX_METRIC_CALLS,
        "num_candidates": len(result.candidates) if hasattr(result, 'candidates') else 0,
    }
    
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    
    print(f"\n✓ Results saved to {output_dir}/")
    print(f"  Best program: {output_dir}/best_program.py")
    print(f"  Best score: {summary['best_score']:.4f}")


# ================================================================
# MAIN
# ================================================================

def main():
    print("\n" + "=" * 70)
    print("  GEPA PROGRAM-BASED OPTIMIZATION")
    print("  Telugu Harmful Speech Classification")
    print("=" * 70 + "\n")
    
    # Check API key
    print("[1/6] Checking API key...")
    api_key = check_api_key()
    
    # Setup DSPy LM
    print(f"\n[2/6] Setting up DSPy LM...")
    print(f"      Task model: {TASK_MODEL}")
    print(f"      Reflection model: {REFLECTION_MODEL}")
    
    lm = dspy.LM(
        model=TASK_MODEL,
        api_base="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=0.3,
        max_tokens=2000
    )
    
    # Set as default
    dspy.configure(lm=lm)
    
    # Load dataset
    print(f"\n[3/6] Loading dataset from {CSV_PATH}...")
    trainset, valset = load_dataset(CSV_PATH, SAMPLE_SIZE, VAL_SIZE)
    print(f"      Trainset: {len(trainset)} examples")
    print(f"      Valset: {len(valset)} examples")
    
    # Load seed program
    print(f"\n[4/6] Loading seed program...")
    seed_program_src = load_seed_program()
    print(f"      Program length: {len(seed_program_src):,} characters")
    print(f"      Program lines: {seed_program_src.count(chr(10))} lines")
    
    # Create seed candidate
    seed_candidate = {"program": seed_program_src}
    
    # Initialize adapter
    print(f"\n[5/6] Initializing adapter...")
    adapter = TeluguHarmfulSpeechAdapter(lm_for_task=lm)
    
    # Validate seed program works
    print(f"      Validating seed program on 3 examples...")
    try:
        test_batch = adapter.evaluate(
            dataset=trainset[:3],
            candidate=seed_candidate,
            capture_traces=True
        )
        avg_score = sum(test_batch.scores) / len(test_batch.scores)
        print(f"      ✓ Seed program works! Avg score: {avg_score:.2f}")
    except Exception as e:
        print(f"      ✗ Seed program failed: {e}")
        sys.exit(1)
    
    # Run GEPA
    print(f"\n[6/6] Running GEPA optimization...")
    print(f"      Budget: {MAX_METRIC_CALLS} metric calls")
    print(f"      This will take approximately {MAX_METRIC_CALLS * 0.5 / 60:.0f}-{MAX_METRIC_CALLS * 1.5 / 60:.0f} minutes")
    print()
    
    import time
    t0 = time.time()
    
    result = optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=None,
        max_metric_calls=MAX_METRIC_CALLS,
        display_progress_bar=True,
    )



    
    elapsed = time.time() - t0
    
    print(f"\n⏱ Optimization finished in {elapsed / 60:.1f} minutes")
    
    # Save results
    save_results(result, OUTPUT_DIR)
    
    # Print best program preview
    best_program = result.best_candidate["program"]
    print("\n" + "=" * 70)
    print("BEST EVOLVED PROGRAM (first 1000 chars)")
    print("=" * 70)
    print(best_program[:1000])
    print("...")
    print("=" * 70)
    
    print(f"\nValidation scores over iterations: {result.val_aggregate_scores}")
    print(f"Best score: {max(result.val_aggregate_scores):.4f}")
    print(f"\nFull program: {OUTPUT_DIR}/best_program.py")
    print("\n✓ Done!")


if __name__ == "__main__":
    main()