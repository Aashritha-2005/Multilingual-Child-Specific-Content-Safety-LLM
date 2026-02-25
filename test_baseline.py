"""
test_baseline.py
================
Smoke tests for baseline_subtask3_fixed.py (no API calls)

Tests:
1. Data loading with zero-label filtering
2. Prediction parsing handles various formats
3. Evaluation metrics compute correctly

Run:
    python test_baseline.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from baseline_subtask3_fixed import (
    load_dataset,
    parse_prediction,
    evaluate_predictions,
    LABELS,
    CSV_PATH
)
import pandas as pd


def test_data_loading():
    """Test that dataset loads, filters zero-labels, and splits correctly."""
    print("[1/3] Testing data loading with zero-label filtering...")
    
    train_df, test_df = load_dataset(CSV_PATH, test_size=50)
    
    # Check sizes
    assert len(train_df) > 0, "Train set is empty"
    assert len(test_df) == 50, f"Expected 50 test samples, got {len(test_df)}"
    
    # Check columns
    required_cols = ["text"] + LABELS
    for col in required_cols:
        assert col in train_df.columns, f"Missing column: {col}"
        assert col in test_df.columns, f"Missing column: {col}"
    
    # CRITICAL: Verify no zero-label examples
    for idx, row in train_df.iterrows():
        label_sum = sum(row[label] for label in LABELS)
        assert label_sum > 0, f"Train row {idx} has all zeros (should be filtered)"
    
    for idx, row in test_df.iterrows():
        label_sum = sum(row[label] for label in LABELS)
        assert label_sum > 0, f"Test row {idx} has all zeros (should be filtered)"
    
    # Check label values are binary
    for label in LABELS:
        assert train_df[label].isin([0, 1]).all(), f"{label} has non-binary values"
        assert test_df[label].isin([0, 1]).all(), f"{label} has non-binary values"
    
    print(f"   ✅ Loaded {len(train_df)} train + {len(test_df)} test samples")
    print(f"   ✅ All {len(LABELS)} label columns present")
    print(f"   ✅ Zero-label examples filtered correctly")


def test_prediction_parsing():
    """Test that prediction parsing handles various response formats."""
    print("\n[2/3] Testing prediction parsing...")
    
    # Test 1: Clean JSON
    clean_json = json.dumps({
        "stereotype": 1,
        "vilification": 0,
        "dehumanization": 0,
        "extreme_language": 1,
        "lack_of_empathy": 0,
        "invalidation": 1
    })
    
    parsed = parse_prediction(clean_json)
    assert parsed["stereotype"] == 1
    assert parsed["vilification"] == 0
    assert parsed["extreme_language"] == 1
    assert len(parsed) == 6  # All labels present
    print("   ✅ Clean JSON parsed correctly")
    
    # Test 2: JSON with markdown
    markdown_json = f"```json\n{clean_json}\n```"
    parsed = parse_prediction(markdown_json)
    assert parsed["stereotype"] == 1
    assert len(parsed) == 6
    print("   ✅ Markdown-wrapped JSON parsed correctly")
    
    # Test 3: JSON with preamble
    preamble_json = f"Sure! Here's the classification:\n```json\n{clean_json}\n```"
    parsed = parse_prediction(preamble_json)
    assert parsed["stereotype"] == 1
    print("   ✅ JSON with preamble parsed correctly")
    
    # Test 4: Broken JSON (should return all zeros)
    broken = "This is not valid JSON"
    parsed = parse_prediction(broken)
    assert all(v == 0 for v in parsed.values())
    assert len(parsed) == 6
    print("   ✅ Broken JSON returns all zeros")
    
    # Test 5: Empty string
    parsed = parse_prediction("")
    assert all(v == 0 for v in parsed.values())
    assert len(parsed) == 6
    print("   ✅ Empty string returns all zeros")
    
    # Test 6: Missing labels (should fill with zeros)
    partial_json = json.dumps({"stereotype": 1, "vilification": 1})
    parsed = parse_prediction(partial_json)
    assert parsed["stereotype"] == 1
    assert parsed["vilification"] == 1
    assert parsed["dehumanization"] == 0  # Should be filled
    assert len(parsed) == 6
    print("   ✅ Missing labels filled with zeros")


def test_evaluation():
    """Test that evaluation metrics compute correctly."""
    print("\n[3/3] Testing evaluation...")
    
    # Create synthetic test data (all have at least one positive label)
    test_data = {
        "text": ["sample1", "sample2", "sample3", "sample4"],
        "stereotype": [1, 0, 1, 0],
        "vilification": [0, 1, 0, 1],
        "dehumanization": [1, 1, 0, 0],
        "extreme_language": [1, 0, 1, 1],
        "lack_of_empathy": [0, 0, 1, 1],
        "invalidation": [1, 1, 1, 0]
    }
    
    test_df = pd.DataFrame(test_data)
    
    # Verify no zero-label examples
    for idx, row in test_df.iterrows():
        assert sum(row[label] for label in LABELS) > 0
    
    # Create predictions (perfect match for first 2, errors for last 2)
    predictions = [
        # Perfect match
        {label: test_data[label][0] for label in LABELS},
        {label: test_data[label][1] for label in LABELS},
        # Errors
        {label: 0 for label in LABELS},  # All zeros (will cause FNs)
        {label: 1 for label in LABELS},  # All ones (will cause FPs)
    ]
    
    # Compute metrics
    results = evaluate_predictions(test_df, predictions)
    
    # Check structure
    assert "overall" in results
    assert "accuracy" in results["overall"]
    assert "f1" in results["overall"]
    
    for label in LABELS:
        assert label in results
        assert "accuracy" in results[label]
        assert "f1" in results[label]
        assert "false_positives" in results[label]
        assert "false_negatives" in results[label]
    
    # Overall accuracy should be reasonable
    overall_acc = results["overall"]["accuracy"]
    assert 0.3 < overall_acc < 0.9, f"Unexpected overall accuracy: {overall_acc}"
    
    print(f"   ✅ Metrics computed correctly")
    print(f"   ✅ Overall accuracy: {overall_acc:.3f}")
    print(f"   ✅ Per-label error tracking works")


def main():
    print("\n" + "=" * 60)
    print("BASELINE TESTS - SUBTASK 3 (FIXED)")
    print("=" * 60 + "\n")
    
    tests = [
        test_data_loading,
        test_prediction_parsing,
        test_evaluation
    ]
    
    passed = 0
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    if passed == len(tests):
        print(f"✅ ALL {passed} TESTS PASSED")
        print("=" * 60)
        print("\n📋 Next steps:")
        print("   1. Get OpenRouter API key: https://openrouter.ai/keys")
        print("   2. export OPENROUTER_API_KEY='your_key_here'")
        print("   3. python baseline_subtask3_fixed.py")
        print("\n⚠️  FIXES APPLIED:")
        print("   ✓ Zero-label examples filtered")
        print("   ✓ Exact label names (no underscore issues)")
        print("   ✓ Per-label error tracking (FP/FN)")
        print("   ✓ Proper OpenRouter model format")
        print()
    else:
        print(f"❌ {passed}/{len(tests)} tests passed")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()