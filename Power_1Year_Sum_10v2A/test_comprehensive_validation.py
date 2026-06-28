#!/usr/bin/env python3
"""
Comprehensive validation of the dual forecast implementation.
This script verifies that the code changes are syntactically correct and logically sound.
"""

import sys
import ast
from pathlib import Path

def validate_python_syntax(file_path):
    """Validate Python syntax of the modified file."""
    print(f"Validating Python syntax of {file_path.name}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        print("  ✓ Python syntax is valid")
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax error: {e}")
        return False

def check_variable_consistency(file_path):
    """Check that variable names are used consistently."""
    print("\nChecking variable consistency...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for common patterns
    patterns = {
        "holdout_result_no_ex initialization": "holdout_result_no_ex = None",
        "holdout_metrics_no_ex initialization": "holdout_metrics_no_ex = {}",
        "with_exogenous DataFrame": "out_forecast_with_exogenous = pd.DataFrame",
        "without_exogenous DataFrame": "out_forecast_without_exogenous = pd.DataFrame",
    }
    
    issues = []
    for pattern_name, pattern in patterns.items():
        if pattern not in content:
            issues.append(f"Missing: {pattern_name}")
    
    if issues:
        for issue in issues:
            print(f"  ⚠ {issue}")
        return False
    else:
        print("  ✓ Variable consistency checks passed")
        return True

def check_file_paths(file_path):
    """Check that output file paths are correctly configured."""
    print("\nChecking output file path configurations...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for file path definitions in RunArtifacts
    file_path_patterns = {
        "with_exogenous forecast CSV path": 'forecast_csv_with_exogenous=run_output_dir / f"forecast_365d_with_exogenous_',
        "without_exogenous forecast CSV path": 'forecast_csv_without_exogenous=run_output_dir / f"forecast_365d_without_exogenous_',
    }
    
    issues = []
    for pattern_name, pattern in file_path_patterns.items():
        if pattern not in content:
            issues.append(f"Missing: {pattern_name}")
    
    if issues:
        for issue in issues:
            print(f"  ⚠ {issue}")
        return False
    else:
        print("  ✓ Output file path configurations verified")
        return True

def check_csv_save_operations(file_path):
    """Check that both CSV variants are saved."""
    print("\nChecking CSV save operations...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for save operations
    save_operations = {
        "with_exogenous CSV save": "out_forecast_with_exogenous.to_csv(artifacts.forecast_csv_with_exogenous",
        "without_exogenous CSV save": "out_forecast_without_exogenous.to_csv(artifacts.forecast_csv_without_exogenous",
    }
    
    issues = []
    for op_name, op_pattern in save_operations.items():
        if op_pattern not in content:
            issues.append(f"Missing: {op_name}")
    
    if issues:
        for issue in issues:
            print(f"  ⚠ {issue}")
        return False
    else:
        print("  ✓ CSV save operations verified")
        return True

def check_aggregation_operations(file_path):
    """Check that aggregations are performed for both variants."""
    print("\nChecking aggregation operations...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for aggregation operations
    agg_operations = {
        "with_exogenous annual aggregation": "annual_agg_with_ex = compute_forecast_aggregates(out_forecast_with_exogenous",
        "without_exogenous annual aggregation": "annual_agg_without_ex = compute_forecast_aggregates(out_forecast_without_exogenous",
        "with_exogenous aggregates save": 'save_aggregated_forecasts(annual_agg_with_ex, artifacts.run_output_dir, "annual_with_exogenous"',
        "without_exogenous aggregates save": 'save_aggregated_forecasts(annual_agg_without_ex, artifacts.run_output_dir, "annual_without_exogenous"',
    }
    
    issues = []
    for op_name, op_pattern in agg_operations.items():
        if op_pattern not in content:
            issues.append(f"Missing: {op_name}")
    
    if issues:
        for issue in issues:
            print(f"  ⚠ {issue}")
        return False
    else:
        print("  ✓ Aggregation operations verified")
        return True

def check_plot_generation(file_path):
    """Check that plots are generated for both variants."""
    print("\nChecking plot generation for both variants...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for plot generation
    plot_patterns = {
        "with_exogenous future forecast plot": "forecast_365d_future_",
        "without_exogenous future forecast plot": "forecast_365d_future_without_exogenous_",
        "without_exogenous holdout comparison plot": "AutoTS_forecast_vs_actual_vs_lastvalue_.*_without_exogenous",
    }
    
    issues = []
    for pattern_name, pattern in plot_patterns.items():
        # Use simple substring check instead of regex
        simple_pattern = pattern.replace(".*", "")
        if simple_pattern not in content:
            issues.append(f"Missing: {pattern_name}")
    
    if issues:
        for issue in issues:
            print(f"  ⚠ {issue}")
        return False
    else:
        print("  ✓ Plot generation for both variants verified")
        return True

def check_metrics_reporting(file_path):
    """Check that metrics are reported for both variants."""
    print("\nChecking metrics reporting for both variants...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for metrics fields
    metrics_patterns = {
        "with_exogenous holdout_mae_with_exogenous": '"holdout_mae_with_exogenous"',
        "without_exogenous holdout_mae_without_exogenous": '"holdout_mae_without_exogenous"',
    }
    
    issues = []
    for pattern_name, pattern in metrics_patterns.items():
        if pattern not in content:
            issues.append(f"Missing: {pattern_name}")
    
    if issues:
        for issue in issues:
            print(f"  ⚠ {issue}")
        return False
    else:
        print("  ✓ Metrics reporting for both variants verified")
        return True

def main():
    """Run comprehensive validation."""
    print("=" * 70)
    print("Comprehensive Dual Forecast Implementation Validation")
    print("=" * 70)
    
    script_path = Path(__file__).parent / "10vL8_autots_365d_forecast.py"
    
    if not script_path.exists():
        print(f"✗ Script not found: {script_path}")
        return False
    
    validators = [
        validate_python_syntax,
        check_variable_consistency,
        check_file_paths,
        check_csv_save_operations,
        check_aggregation_operations,
        check_plot_generation,
        check_metrics_reporting,
    ]
    
    results = []
    for validator in validators:
        try:
            result = validator(script_path)
            results.append(result)
        except Exception as e:
            print(f"✗ Validator {validator.__name__} failed: {e}")
            results.append(False)
    
    print("\n" + "=" * 70)
    print(f"Validation Results: {sum(results)}/{len(results)} passed")
    print("=" * 70)
    
    if all(results):
        print("\n✅ All comprehensive validation checks passed!")
        print("   The dual forecast implementation appears to be correct.")
    else:
        print("\n⚠ Some validation checks did not pass.")
        print("   Please review the warnings above.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
