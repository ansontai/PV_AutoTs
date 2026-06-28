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
from datetime import datetime
from pathlib import Path

# DEFAULT_FORECAST_CSV = 'Power_1Year_Sum_10v2A\output\10vL6_autots_365d_forecast_20260513_161911\forecast_365d_20260513_161911.csv'
DEFAULT_ENABLE_ONLY_DOC_HANDLE = False

DEFAULT_SEEDS_FILE_NAME = 'autots_seeds_20260420_034832-1seed.csv'
# DEFAULT_SEEDS_FILE_NAME = 'autots_seeds_20260420_034832-30seed.csv'

DEFAULT_FIT_FORECAST_LENGTH = 120
# DEFAULT_FIT_FORECAST_LENGTH = 90
# DEFAULT_FIT_FORECAST_LENGTH = 3
# DEFAULT_FIT_FORECAST_LENGTH = 2

# DEFAULT_MAX_GENERATIONS = 50
DEFAULT_MAX_GENERATIONS = 1

# DEFAULT_SEEDS_START_INDEX = "auto"  # can be 'auto' to auto-resume from latest effective_settings*.json seed, or numeric (None/0/1=first seed, 2=second seed, etc.)
DEFAULT_SEEDS_START_INDEX = 0  # start from first seed by default, can be overridden by CLI
# DEFAULT_SEEDS_START_INDEX = 13
# DEFAULT_SEEDS_START_INDEX = "auto"

# 時間特徵開關 (False=物理特徵只, True=物理+時間特徵)
INCLUDE_TEMPORAL_FEATURES = False

# TMY 外生變數開關預設 (True = 使用 TMY；可由 CLI 以 --no_tmy_exogenous 關閉)
# INCLUDE_TMY_EXOGENOUS = False
INCLUDE_TMY_EXOGENOUS = True

DEFAULT_OUTPUT_TAG = Path(__file__).resolve().stem.replace('10v', '')


def build_child_command(
    python_exe: str,
    child_script: Path,
    child_output_root: Path,
    fit_forecast_length: str,
    max_generations: int,
    include_temporal_features: bool,
    doc_only: bool,
    saved_forecast_csv: str | None,
    random_seed: int,
    no_tmy_exogenous: bool,
    output_mode_tag: str | None,
    loop: bool,
) -> list[str]:
    cmd = [python_exe, str(child_script), '--output_dir', str(child_output_root)]
    if output_mode_tag:
        cmd += ['--output_mode_tag', str(output_mode_tag)]
    if include_temporal_features:
        cmd += ['--include_temporal_features']
    if no_tmy_exogenous:
        cmd += ['--no_tmy_exogenous']
    if doc_only:
        cmd += ['--doc-only']
    if saved_forecast_csv:
        cmd += ['--saved-forecast-csv', saved_forecast_csv]
    cmd += ['--fit_forecast_length', str(fit_forecast_length), '--max_generations', str(max_generations)]
    cmd += ['--random_seed', str(int(random_seed))]
    if loop:
        cmd += ['--loop']
    return cmd

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
    parser = argparse.ArgumentParser(description='Launcher for 10vL10 365d forecast')
    parser.add_argument('--seeds_file', default=None, help='Path or name of seeds CSV')
    parser.add_argument('--seeds_start_index', default=DEFAULT_SEEDS_START_INDEX, help="'auto' or numeric index (1-based semantics)")
    parser.add_argument('--output_dir', default='.', help='Root output dir for child runs')
    parser.add_argument('--output_tag', default=DEFAULT_OUTPUT_TAG, help='Tag appended under output_dir')
    parser.add_argument('--loop', action='store_true', help='Forward loop to child')
    parser.add_argument('--fit_forecast_length', default=str(DEFAULT_FIT_FORECAST_LENGTH), help="Override FIT_FORECAST_LENGTH for child (numeric or 'auto')")
    parser.add_argument('--max_generations', type=int, default=DEFAULT_MAX_GENERATIONS, help='Override MAX_GENERATIONS for child')
    parser.add_argument('--no_tmy_exogenous', action='store_true', help='Forward to child to disable using TMY as future exogenous regressors')
    parser.add_argument('--run_mode', choices=['both', 'with_exogenous', 'without_exogenous'], default='both', help='Select whether to run both modes or only one mode')
    parser.add_argument('--doc-only', action='store_true', default=False, help='Skip AutoTS fit/predict and use saved forecast CSV for doc/chart regeneration')
    parser.add_argument('--saved-forecast-csv', default=None, help='Path to saved forecast CSV for doc-only mode')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    child_script = script_dir / '10vL10b_autots_365d_forecast.py'
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

    requested_mode = args.run_mode
    if args.no_tmy_exogenous and requested_mode == 'both':
        requested_mode = 'without_exogenous'

    mode_specs: list[tuple[str, bool]]
    if requested_mode == 'both':
        mode_specs = [
            ('with_exogenous', False),
            ('without_exogenous', True),
        ]
    elif requested_mode == 'with_exogenous':
        mode_specs = [('with_exogenous', False)]
    else:
        mode_specs = [('without_exogenous', True)]

    def run_one_mode(mode_tag: str, no_tmy_exogenous: bool) -> None:
        # child will create timestamped run dir under child_output_root and include mode_tag
        mode_output_root = child_output_root
        mode_output_root.mkdir(parents=True, exist_ok=True)

        if seeds:
            start_index = _resolve_seeds_start_index(args.seeds_start_index, seeds, mode_output_root)
            if start_index is None:
                start_index = 0
            for enum_index, s in enumerate(seeds[start_index:], start=start_index):
                actual_position = enum_index + 1
                cmd = build_child_command(
                    python_exe=python_exe,
                    child_script=child_script,
                    child_output_root=mode_output_root,
                    fit_forecast_length=args.fit_forecast_length,
                    max_generations=args.max_generations,
                    include_temporal_features=INCLUDE_TEMPORAL_FEATURES,
                    doc_only=args.doc_only,
                    saved_forecast_csv=args.saved_forecast_csv,
                    random_seed=int(s),
                    no_tmy_exogenous=no_tmy_exogenous,
                    output_mode_tag=mode_tag,
                    loop=args.loop,
                )
                print(f'Running {mode_tag} seed {actual_position}/{len(seeds)}:', ' '.join(map(str, cmd)))
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f'Child process failed for {mode_tag} seed {actual_position}/{len(seeds)}; continuing. Error: {e}')
                    continue
        else:
            cmd = build_child_command(
                python_exe=python_exe,
                child_script=child_script,
                child_output_root=mode_output_root,
                fit_forecast_length=args.fit_forecast_length,
                max_generations=args.max_generations,
                include_temporal_features=INCLUDE_TEMPORAL_FEATURES,
                doc_only=args.doc_only,
                saved_forecast_csv=args.saved_forecast_csv,
                random_seed=0,
                no_tmy_exogenous=no_tmy_exogenous,
                output_mode_tag=mode_tag,
                loop=args.loop,
            )
            print(f'Running single child for {mode_tag}:', ' '.join(map(str, cmd)))
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f'Child process failed for single run ({mode_tag}); exiting.', e)
                sys.exit(getattr(e, 'returncode', 1))

    for mode_tag, no_tmy_exogenous in mode_specs:
        run_one_mode(mode_tag=mode_tag, no_tmy_exogenous=no_tmy_exogenous)

# After running modes, attempt to summarize holdout/settings for each mode into one CSV
try:
    summary_rows = []
    for mode_tag, _ in mode_specs:
            # look for latest run directory under child_output_root/<child>/<seed>/<timestamp>/<mode>
            child_run_root = child_output_root / child_script.stem
            candidates = sorted(
                [p for p in child_run_root.rglob('effective_settings_*.json') if p.parent.name == mode_tag],
                key=lambda p: p.stat().st_mtime if p.exists() else 0,
                reverse=True,
            )
        chosen_settings = None
            if candidates:
                chosen_settings = candidates[0]
        if chosen_settings is None:
            # no settings found for this mode
            continue
        try:
            with open(chosen_settings, 'r', encoding='utf-8') as f:
                settings = json.load(f)
        except Exception:
            continue
        row = {
            'mode': mode_tag,
            'effective_settings': str(chosen_settings),
            'output_dir': str(chosen_settings.parent),
            'timestamp': settings.get('timestamp', ''),
            'random_seed': settings.get('random_seed', ''),
            'use_tmy_exogenous': settings.get('use_tmy_exogenous', ''),
            'autots_score': settings.get('autots_score', ''),
            'validation_smape': settings.get('validation_smape', ''),
            'holdout_mae': settings.get('holdout_mae', ''),
            'holdout_rmse': settings.get('holdout_rmse', ''),
            'holdout_mape_pct': settings.get('holdout_mape_pct', ''),
            'holdout_mase': settings.get('holdout_mase', ''),
            'holdout_length': settings.get('holdout_length', ''),
            'best_model': settings.get('best_model', ''),
        }
        summary_rows.append(row)

    if summary_rows:
        summary_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = child_output_root / f"ab_holdout_summary_{summary_ts}.csv"
        with open(summary_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for r in summary_rows:
                writer.writerow(r)
        print(f'Launcher: wrote A/B summary CSV: {summary_file}')
    else:
        print('Launcher: no summary rows found (no effective_settings JSONs located)')
except Exception as e:
    print(f'Launcher: failed to write A/B summary CSV: {e}')
