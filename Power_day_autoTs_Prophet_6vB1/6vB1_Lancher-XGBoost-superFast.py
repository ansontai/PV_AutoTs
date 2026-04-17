#!/usr/bin/env python3
"""Launch helper for 6vB1-autoTs_WeatherToDayWh.py"""
import argparse
import subprocess
import os
import sys

# Default settings (can be changed here)
# DEFAULT_HORIZONS = [3, 6, 9]
DEFAULT_HORIZONS = [2, 3, 4]
DEFAULT_N_JOBS = -1
DEFAULT_MAX_GENERATIONS = 1
DEFAULT_TRANSFORMER_LIST = 'default'
DEFAULT_MODEL_LIST = ['XGBoost']
DEFAULT_ENSEMBLE = None
DEFAULT_INPUT_FILE = None
DEFAULT_LOOP = False
# DEFAULT_LOOP = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch AutoTS script with custom parameters')
    parser.add_argument('--horizons', nargs='+', type=int, default=DEFAULT_HORIZONS)
    parser.add_argument('--n_jobs', type=int, default=DEFAULT_N_JOBS)
    parser.add_argument('--max_generations', type=int, default=DEFAULT_MAX_GENERATIONS)
    parser.add_argument('--transformer_list', type=str, default=DEFAULT_TRANSFORMER_LIST)
    parser.add_argument('--model_list', nargs='+', default=DEFAULT_MODEL_LIST)
    parser.add_argument('--ensemble', default=DEFAULT_ENSEMBLE)
    parser.add_argument('--input_file', default=DEFAULT_INPUT_FILE)
    parser.add_argument('--loop', action='store_true', default=DEFAULT_LOOP, help='Repeat the script until interrupted')
    args = parser.parse_args()

    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '6vB1-autoTs_WeatherToDayWh.py'))

    cmd = [
        sys.executable,
        script_path,
        '--horizons', *map(str, args.horizons),
        '--n_jobs', str(args.n_jobs),
        '--max_generations', str(args.max_generations),
        '--transformer_list', args.transformer_list,
        '--model_list', *args.model_list,
    ]
    if args.loop:
        cmd += ['--loop']
    if args.ensemble is not None:
        cmd += ['--ensemble', str(args.ensemble)]
    if args.input_file:
        cmd += ['--input_file', args.input_file]

    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
