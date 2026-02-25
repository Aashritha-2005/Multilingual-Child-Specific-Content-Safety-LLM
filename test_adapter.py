import sys
import os
from gepa import EvaluationBatch

from run_gepa import SemEvalGEPAAdapter, CSV_PATH, SEED_PROMPT

PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")

def test_adapter():
    
    print("\n" + "="*50)
    print("GEPA ADAPTER TEST SUITE")
    print("="*50 + "\n")

    # Test 1: API Key
    print("[1/5] Checking API key...")
    if not PERPLEXITY_API_KEY:
        print("❌ PERPLEXITY_API_KEY not set!")
        print("   Set it with: export PERPLEXITY_API_KEY='your-key'")
        return False
    print("✓ API key is set")

    # Test 2: Adapter initialization
    print("\n[2/5] Initializing adapter...")
    try:
        adapter = SemEvalGEPAAdapter(
            csv_path=CSV_PATH,
            api_key=PERPLEXITY_API_KEY,
            sample_size=3
        )
        print(f"✓ Adapter initialized with {len(adapter.df)} examples")
    except Exception as e:
        print(f"❌ Failed to initialize adapter: {e}")
        return False

    # Test 3: Evaluate WITHOUT traces (standard mode)
    print("\n[3/5] Testing evaluate() without traces...")
    data = adapter.df.to_dict("records")
    
    try:
        result = adapter.evaluate(data, {"system_prompt": SEED_PROMPT}, capture_traces=False)
        
        if not isinstance(result, EvaluationBatch):
            print(f"❌ Wrong return type: {type(result)}")
            return False
        
        if result.trajectories is not None:
            print("❌ Trajectories should be None when capture_traces=False")
            return False
            
        print("✓ evaluate() returns EvaluationBatch")
        print(f"  - {len(result.outputs)} outputs")
        print(f"  - {len(result.scores)} scores")
        print(f"  - Average score: {sum(result.scores)/len(result.scores):.3f}")
    except Exception as e:
        print(f"❌ evaluate() failed: {e}")
        return False

    # Test 4: Evaluate WITH traces (reflection mode)
    print("\n[4/5] Testing evaluate() with traces...")
    
    try:
        result_with_traces = adapter.evaluate(
            data, 
            {"system_prompt": SEED_PROMPT}, 
            capture_traces=True  # ← CRITICAL: Must be True for reflection
        )
        
        if not isinstance(result_with_traces, EvaluationBatch):
            print(f"❌ Wrong return type: {type(result_with_traces)}")
            return False
        
        if result_with_traces.trajectories is None:
            print("❌ Trajectories should NOT be None when capture_traces=True")
            return False
        
        if len(result_with_traces.trajectories) != len(data):
            print(f"❌ Wrong number of trajectories: {len(result_with_traces.trajectories)} != {len(data)}")
            return False
            
        print("✓ evaluate() with traces OK")
        print(f"  - {len(result_with_traces.trajectories)} trajectories captured")
        
        # Check trajectory structure
        sample_traj = result_with_traces.trajectories[0]
        required_keys = ["text", "raw_response", "prediction", "ground_truth"]
        missing = [k for k in required_keys if k not in sample_traj]
        if missing:
            print(f"❌ Trajectories missing keys: {missing}")
            return False
        
        print("✓ Trajectory structure is correct")
        
    except Exception as e:
        print(f"❌ evaluate() with traces failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Reflective dataset creation
    print("\n[5/5] Testing make_reflective_dataset()...")
    
    try:
        refl = adapter.make_reflective_dataset(
            {"system_prompt": SEED_PROMPT}, 
            result_with_traces,  # ← Use result WITH traces
            ["system_prompt"]
        )
        
        if not isinstance(refl, dict):
            print(f"❌ Wrong return type: {type(refl)}")
            return False
        
        if "system_prompt" not in refl:
            print("❌ Missing 'system_prompt' key in reflective dataset")
            return False
        
        if not isinstance(refl["system_prompt"], list):
            print("❌ 'system_prompt' should be a list")
            return False
        
        print("✓ make_reflective_dataset() OK")
        print(f"  - Generated {len(refl['system_prompt'])} reflection records")
        
    except Exception as e:
        print(f"❌ make_reflective_dataset() failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # All tests passed!
    print("\n" + "="*50)
    print("✓ ALL TESTS PASSED - ADAPTER IS READY")
    print("="*50)
    print("\nYou can now run:")
    print("  python run_gepa_advanced.py")
    print("  python run_gepa_optimize_fixed.py")
    print("  python run_gepa_template.py")
    print()
    
    return True


if __name__ == "__main__":
    success = test_adapter()
    if not success:
        print("\n❌ TESTS FAILED - Fix errors above before running GEPA")
        sys.exit(1)
    else:
        sys.exit(0)
