#!/usr/bin/env python3
"""
Test script to verify dual forecast functionality (with and without exogenous variables).
This script performs a quick validation of the code structure and paths.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all necessary imports work."""
    print("Testing imports...")
    try:
        import pandas as pd
        import numpy as np
        print("  ✓ pandas and numpy imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import pandas/numpy: {e}")
        return False
    
    try:
        from autots import AutoTS
        print("  ✓ AutoTS imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import AutoTS: {e}")
        return False
    
    return True

def test_dataclass_structure():
    """Test that RunArtifacts dataclass has the new fields."""
    print("\nTesting RunArtifacts dataclass structure...")
    try:
        from pathlib import Path
        from dataclasses import fields
        
        # Try to import the script to get the dataclass definition
        # Since we can't directly import due to script structure, we'll check the file
        script_path = Path(__file__).parent / "10vL8_autots_365d_forecast.py"
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for new fields in RunArtifacts
        required_fields = [
            "forecast_csv_with_exogenous",
            "forecast_csv_without_exogenous",
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in content:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"  ✗ Missing fields in RunArtifacts: {missing_fields}")
            return False
        
        print("  ✓ All required fields present in RunArtifacts")
        return True
    except Exception as e:
        print(f"  ✗ Error checking dataclass structure: {e}")
        return False

def test_prediction_logic():
    """Test that prediction logic has been updated."""
    print("\nTesting prediction logic modifications...")
    try:
        script_path = Path(__file__).parent / "10vL8_autots_365d_forecast.py"
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for key modifications
        checks = [
            ("second prediction without exogenous", "prediction_no_exogenous = model.predict(forecast_length=FORECAST_LENGTH, future_regressor=None)"),
            ("forecast_csv_with_exogenous save", "out_forecast_with_exogenous.to_csv(artifacts.forecast_csv_with_exogenous"),
            ("forecast_csv_without_exogenous save", "out_forecast_without_exogenous.to_csv(artifacts.forecast_csv_without_exogenous"),
        ]
        
        failed_checks = []
        for check_name, check_str in checks:
            if check_str not in content:
                failed_checks.append(check_name)
        
        if failed_checks:
            print(f"  ✗ Missing modifications: {failed_checks}")
            return False
        
        print("  ✓ All prediction logic modifications present")
        return True
    except Exception as e:
        print(f"  ✗ Error checking prediction logic: {e}")
        return False

def test_aggregation_logic():
    """Test that aggregation logic has been updated."""
    print("\nTesting aggregation logic modifications...")
    try:
        script_path = Path(__file__).parent / "10vL8_autots_365d_forecast.py"
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for aggregation updates
        checks = [
            ("with_exogenous aggregation", "annual_agg_with_ex = compute_forecast_aggregates(out_forecast_with_exogenous"),
            ("without_exogenous aggregation", "annual_agg_without_ex = compute_forecast_aggregates(out_forecast_without_exogenous"),
        ]
        
        failed_checks = []
        for check_name, check_str in checks:
            if check_str not in content:
                failed_checks.append(check_name)
        
        if failed_checks:
            print(f"  ✗ Missing aggregation modifications: {failed_checks}")
            return False
        
        print("  ✓ All aggregation logic modifications present")
        return True
    except Exception as e:
        print(f"  ✗ Error checking aggregation logic: {e}")
        return False

def test_metrics_updates():
    """Test that metrics payload has been updated."""
    print("\nTesting metrics payload updates...")
    try:
        script_path = Path(__file__).parent / "10vL8_autots_365d_forecast.py"
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for metrics updates
        checks = [
            ("with_exogenous metrics", "holdout_mae_with_exogenous"),
            ("without_exogenous metrics", "holdout_mae_without_exogenous"),
        ]
        
        failed_checks = []
        for check_name, check_str in checks:
            if check_str not in content:
                failed_checks.append(check_name)
        
        if failed_checks:
            print(f"  ✗ Missing metrics updates: {failed_checks}")
            return False
        
        print("  ✓ All metrics payload updates present")
        return True
    except Exception as e:
        print(f"  ✗ Error checking metrics updates: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Dual Forecast Functionality Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_dataclass_structure,
        test_prediction_logic,
        test_aggregation_logic,
        test_metrics_updates,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test_func.__name__} raised exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
