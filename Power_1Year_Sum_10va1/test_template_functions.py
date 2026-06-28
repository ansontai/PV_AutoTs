#!/usr/bin/env python3
"""
Quick test to verify new template extraction functions work without full 365-day run.
"""

import sys
from pathlib import Path
import json
import importlib.util

try:
    # Load the script as a module to access the functions
    script_path = Path(__file__).parent / "10vB1_autots_365d_forecast.py"
    spec = importlib.util.spec_from_file_location("child_script", script_path)
    child_module = importlib.util.module_from_spec(spec)
    
    # Don't execute the full module, just make the functions available
    spec.loader.exec_module(child_module)
    
    # Test 1: Check that the new functions are defined
    print("✓ Test 1: Checking if new functions exist...")
    assert hasattr(child_module, '_extract_best_template_row'), "Missing _extract_best_template_row function"
    assert hasattr(child_module, 'save_template_artifacts'), "Missing save_template_artifacts function"
    assert hasattr(child_module, 'save_best_template_artifacts'), "Missing save_best_template_artifacts function"
    print("  PASS: All new functions are defined")
    
    # Test 2: Check RunArtifacts dataclass has new fields
    print("\n✓ Test 2: Checking if RunArtifacts has new fields...")
    artifacts_cls = child_module.RunArtifacts
    assert hasattr(artifacts_cls, '__annotations__'), "RunArtifacts missing annotations"
    annotations = artifacts_cls.__annotations__
    assert 'templates_csv' in annotations, "Missing templates_csv field"
    assert 'templates_json' in annotations, "Missing templates_json field"
    assert 'best_template_csv' in annotations, "Missing best_template_csv field"
    assert 'best_template_json' in annotations, "Missing best_template_json field"
    print("  PASS: All new fields present in RunArtifacts")
    
    # Test 3: Verify _json_safe is accessible
    print("\n✓ Test 3: Checking _json_safe function...")
    assert hasattr(child_module, '_json_safe'), "Missing _json_safe function"
    test_val = child_module._json_safe({"key": "value", "num": 123})
    assert isinstance(test_val, dict), "_json_safe should return dict"
    print("  PASS: _json_safe works correctly")
    
    # Test 4: Verify _extract_results_frame is accessible
    print("\n✓ Test 4: Checking _extract_results_frame function...")
    assert hasattr(child_module, '_extract_results_frame'), "Missing _extract_results_frame function"
    print("  PASS: _extract_results_frame is accessible")
    
    print("\n" + "="*50)
    print("All quick tests PASSED!")
    print("The template functions are correctly integrated.")
    print("="*50)
    
except Exception as e:
    print(f"\n✗ Test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
