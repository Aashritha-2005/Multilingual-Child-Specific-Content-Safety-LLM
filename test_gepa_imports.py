"""
test_gepa_imports.py
====================
Test that GEPA imports work correctly.

This verifies the actual GEPA library structure.
"""

import sys

print("Testing GEPA imports...\n")

# Test 1: Import gepa module
try:
    import gepa
    print("✅ import gepa - SUCCESS")
except ImportError as e:
    print(f"❌ import gepa - FAILED: {e}")
    sys.exit(1)

# Test 2: Import EvaluationBatch
try:
    from gepa import EvaluationBatch
    print("✅ from gepa import EvaluationBatch - SUCCESS")
except ImportError as e:
    print(f"❌ from gepa import EvaluationBatch - FAILED: {e}")
    sys.exit(1)

# Test 3: Check gepa.optimize exists
try:
    assert hasattr(gepa, 'optimize'), "gepa.optimize not found"
    print("✅ gepa.optimize exists - SUCCESS")
except AssertionError as e:
    print(f"❌ gepa.optimize - FAILED: {e}")
    sys.exit(1)

# Test 4: Check what's available in gepa
print("\n📋 Available in gepa module:")
available = [item for item in dir(gepa) if not item.startswith('_')]
for item in sorted(available):
    print(f"   - {item}")

# Test 5: Verify WRONG imports don't work (these should fail)
print("\n🚫 Verifying incorrect imports fail (expected):")

try:
    from gepa import GEPA
    print("❌ from gepa import GEPA - UNEXPECTED SUCCESS (should fail!)")
except ImportError:
    print("✅ from gepa import GEPA - Correctly fails (GEPA class doesn't exist)")

try:
    from gepa import Prediction
    print("❌ from gepa import Prediction - UNEXPECTED SUCCESS (should fail!)")
except ImportError:
    print("✅ from gepa import Prediction - Correctly fails (Prediction class doesn't exist)")

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print("\nCORRECT GEPA USAGE:")
print("  ✓ import gepa")
print("  ✓ from gepa import EvaluationBatch")
print("  ✓ Use: gepa.optimize(...)")
print("  ✓ Adapter returns: EvaluationBatch")
print("\nINCORRECT (don't use):")
print("  ✗ from gepa import GEPA (doesn't exist)")
print("  ✗ from gepa import Prediction (doesn't exist)")
print("  ✗ GEPA(...) (not a class)")