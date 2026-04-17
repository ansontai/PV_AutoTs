#!/usr/bin/env python3
"""Launcher to run AutoTS with DatepartRegression + xgboost regressor."""
import os
import sys
import subprocess
import json


def main():

    script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '6vB1-autoTs_WeatherToDayWh.py')
    )

    model_spec = {
        "model": "DatepartRegression",
        "model_params": {
            "regression_model": {
                "model": "RandomForest",
                "model_params": {"n_estimators": 100, "max_depth": 6},
            },
            "datepart_method": "expanded",
        },
    }

    cmd = [
        sys.executable,
        script_path,
        '--horizons', '2', '3', '4',
        '--n_jobs', '-1',
        '--max_generations', '1',
        '--transformer_list', 'default',
        '--model_list', json.dumps(model_spec),
    ]

    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
	main()

