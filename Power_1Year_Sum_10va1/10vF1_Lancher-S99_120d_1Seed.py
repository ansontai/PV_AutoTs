#!/usr/bin/env python3
"""Launcher for 10vA1_autots_365d_forecast.py with seed file support.

Features:
- discover or accept a seeds CSV (first column or 'SEED' header)
- support --seeds_start_index (numeric or 'auto')
- control child output root and output tag
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_SEEDS_FILE_NAME = 'autots_seeds_20260420_034832-1seed.csv'
# DEFAULT_SEEDS_FILE_NAME = 'autots_seeds_20260420_034832-30seed.csv'

DEFAULT_FIT_FORECAST_LENGTH = 120

# DEFAULT_MAX_GENERATIONS = 1
DEFAULT_MAX_GENERATIONS = 50

# DEFAULT_SEEDS_START_INDEX = "auto"  # can be 'auto' to auto-resume from latest effective_settings*.json seed, or numeric (None/0/1=first seed, 2=second seed, etc.)
DEFAULT_SEEDS_START_INDEX = 0  # start from first seed by default, can be overridden by CLI
# DEFAULT_SEEDS_START_INDEX = 13
# DEFAULT_SEEDS_START_INDEX = "auto"

# 時間特徵開關 (False=物理特徵只, True=物理+時間特徵)
INCLUDE_TEMPORAL_FEATURES = False

DEFAULT_OUTPUT_TAG = Path(__file__).resolve().stem.replace('10vF1_Lancher-', '')


def find_latest_effective_settings(output_root: Path):
    """Find the most recent effective_settings*.json in output tree (for auto-resume)."""
    try:
        candidates = list(Path(output_root).rglob('effective_settings*.json'))
    except Exception:
        return None
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def find_seeds_file(start_path: Path, filename: str | None):
    parents = [start_path, *start_path.parents]
    if filename is None:
        for p in parents:
            at = p / 'AutoTs_SeedGen'
            if at.exists() and at.is_dir():
                for f in sorted(at.glob('autots_seeds*.csv')):
                    return str(f)
            inp = p / 'input'
            if inp.exists() and inp.is_dir():
                for f in sorted(inp.glob('autots_seeds*.csv')):
                    return str(f)
        return None

    # filename provided: if path-like, try as-is or relative
    try:
        fpath = Path(filename)
    except Exception:
        fpath = None
    if fpath and (fpath.is_absolute() or ('/' in filename) or ('\\' in filename) or (os.sep in filename)):
        cand = fpath
        if not cand.is_absolute():
            cand = (start_path / filename).resolve()
        if cand.exists():
            return str(cand)
        cand2 = Path.cwd() / filename
        if cand2.exists():
            return str(cand2)
        return None

    for p in parents:
        cand = p / 'AutoTs_SeedGen' / filename
        if cand.exists():
            return str(cand)
        cand2 = p / 'input' / filename
        if cand2.exists():
            return str(cand2)
    return None


def load_seeds(seeds_path: str):
    seeds = []
    try:
        with open(seeds_path, newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                key = None
                for k in reader.fieldnames:
                    if k.upper() == 'SEED':
                        key = k
                        break
                if key:
                    for row in reader:
                        val = row.get(key)
                        if val is None:
                            continue
                        val = str(val).strip()
                        if val:
                            try:
                                seeds.append(int(val))
                            except ValueError:
                                continue
                    return seeds
            f.seek(0)
            for row in csv.reader(f):
                if not row:
                    continue
                v = row[0].strip()
                try:
                    seeds.append(int(v))
                except ValueError:
                    continue
    except Exception:
        return []
    return seeds


def _resolve_seeds_start_index(raw_value, seeds, output_root):
    """Resolve seeds_start_index parameter, supporting both numeric and 'auto' modes.
    
    Args:
        raw_value: CLI value (str, int, or None)
        seeds: list of seed integers loaded from CSV
        output_root: Path to output directory for finding latest effective_settings*.json
    
    Returns:
        Normalized 0-based index to start from, or None on error
        
    Raises:
        SystemExit on unrecoverable error (seed not found, latest_settings invalid)
    """
    # Handle auto mode
    if isinstance(raw_value, str) and raw_value.strip().lower() == 'auto':
        latest_settings = find_latest_effective_settings(Path(output_root))
        if not latest_settings:
            print("Launcher: --seeds_start_index=auto but no effective_settings*.json found in output tree; aborting.")
            sys.exit(1)
        
        try:
            with open(latest_settings, 'r', encoding='utf-8') as f:
                settings_json = json.load(f)
            latest_seed = int(settings_json.get('random_seed'))
        except Exception as e:
            print(f"Launcher: failed to read or parse random_seed from {latest_settings}: {e}; aborting.")
            sys.exit(1)
        
        # Find index of latest_seed in seeds list
        try:
            found_index = seeds.index(latest_seed)
        except ValueError:
            print(f"Launcher: random_seed {latest_seed} from latest effective_settings*.json not found in seeds list; aborting.")
            sys.exit(1)
        
        # Compute next index
        next_index = found_index + 1
        
        # Check if next seed exists
        if next_index >= len(seeds):
            print(f"Launcher: random_seed {latest_seed} is the last seed in list (index {found_index}/{len(seeds)-1}); no next seed to run.")
            sys.exit(0)
        
        print(f'Launcher: auto mode - found latest seed {latest_seed} at index {found_index}, resuming from next seed at index {next_index} (seed #{next_index + 1}/{len(seeds)})')
        return next_index
    
    # Handle numeric mode
    try:
        if raw_value is None or raw_value == '' or str(raw_value).strip().lower() == 'none':
            normalized = 0
        else:
            num_val = int(raw_value)
            if num_val is None or num_val == 0 or num_val == 1:
                normalized = 0
            elif num_val >= 2:
                normalized = num_val - 1
            else:
                return None
    except (ValueError, TypeError):
        print(f"Launcher: --seeds_start_index must be 'auto' or a numeric value; got '{raw_value}'; aborting.")
        sys.exit(1)
    
    if normalized >= len(seeds):
        print(f"Launcher: seeds_start_index={raw_value} resolves to index {normalized} which is out of range (total seeds={len(seeds)}); aborting.")
        sys.exit(1)
    
    return normalized


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launcher for 10vE1 365d forecast')
    parser.add_argument('--seeds_file', default=None, help='Path or name of seeds CSV')
    parser.add_argument('--seeds_start_index', default=DEFAULT_SEEDS_START_INDEX, help="'auto' or numeric index (1-based semantics)")
    parser.add_argument('--output_dir', default='.', help='Root output dir for child runs')
    parser.add_argument('--output_tag', default=DEFAULT_OUTPUT_TAG, help='Tag appended under output_dir')
    parser.add_argument('--loop', action='store_true', help='Forward loop to child')
    parser.add_argument('--fit_forecast_length', default=str(DEFAULT_FIT_FORECAST_LENGTH), help="Override FIT_FORECAST_LENGTH for child (numeric or 'auto')")
    parser.add_argument('--max_generations', type=int, default=DEFAULT_MAX_GENERATIONS, help='Override MAX_GENERATIONS for child')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    child_script = script_dir / '10vF1_autots_365d_forecast.py'

    # determine child output root
    od = str(args.output_dir).strip()
    if od == '' or od == '.' or od == './':
        child_output_root = (script_dir / 'output' / 'default' / args.output_tag).resolve()
    else:
        cand = Path(od)
        child_output_root = (script_dir / cand).resolve() if not cand.is_absolute() else cand

    print(f'Launcher: using child_output_root={child_output_root}')

    # find seeds file
    seeds_file = None
    if args.seeds_file:
        found = find_seeds_file(script_dir, args.seeds_file)
        if found:
            seeds_file = found
        else:
            p = Path(args.seeds_file)
            if not p.is_absolute():
                p = script_dir / args.seeds_file
            if p.exists():
                seeds_file = str(p)
            else:
                print(f"Launcher: specified seeds_file '{args.seeds_file}' not found; aborting.")
                sys.exit(1)
    else:
        if DEFAULT_SEEDS_FILE_NAME is None:
            found = find_seeds_file(script_dir, None)
            if found:
                seeds_file = found
        else:
            found = find_seeds_file(script_dir, DEFAULT_SEEDS_FILE_NAME)
            if found:
                seeds_file = found
            else:
                print(f"Launcher: default seeds file '{DEFAULT_SEEDS_FILE_NAME}' not found; aborting.")
                sys.exit(1)

    seeds = []
    if seeds_file:
        seeds = load_seeds(seeds_file)
        if not seeds:
            print(f"Launcher: seeds file '{seeds_file}' contains no valid seeds; aborting.")
            sys.exit(1)

    python_exe = sys.executable

    base_cmd = [python_exe, str(child_script), '--output_dir', str(child_output_root)]
    
    # 根據開關添加時間特徵參數
    if INCLUDE_TEMPORAL_FEATURES:
        base_cmd += ['--include_temporal_features']
    # 傳遞 fit_forecast_length 與 max_generations 給子程序
    base_cmd += ['--fit_forecast_length', str(args.fit_forecast_length), '--max_generations', str(args.max_generations)]

    if seeds:
        start_index = _resolve_seeds_start_index(args.seeds_start_index, seeds, child_output_root)
        if start_index is None:
            start_index = 0
        for enum_index, s in enumerate(seeds[start_index:], start=start_index):
            actual_position = enum_index + 1
            cmd = list(base_cmd)
            cmd += ['--random_seed', str(int(s))]
            if args.loop:
                cmd += ['--loop']
            print(f'Running seed {actual_position}/{len(seeds)}:', ' '.join(map(str, cmd)))
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f'Child process failed for seed {actual_position}/{len(seeds)}; continuing. Error: {e}')
                continue
    else:
        cmd = list(base_cmd)
        if args.loop:
            cmd += ['--loop']
        print('Running single child:', ' '.join(map(str, cmd)))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print('Child process failed for single run; exiting.', e)
            sys.exit(getattr(e, 'returncode', 1))
