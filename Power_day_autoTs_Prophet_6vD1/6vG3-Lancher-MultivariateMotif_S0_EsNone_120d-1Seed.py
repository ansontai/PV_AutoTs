#!/usr/bin/env python3
"""Launch helper for 6vD1-autoTs_WeatherToDayWh.py (includes LastValue baseline outputs)."""
import argparse
import subprocess
import os
import sys
from pathlib import Path
import csv
import json
from datetime import datetime

import time
import json

# Default settings (can be changed here)
# DEFAULT_HORIZONS = [2, 3]
# DEFAULT_HORIZONS = [90]
DEFAULT_HORIZONS = [120]
# DEFAULT_HORIZONS = [90, 120]
# DEFAULT_HORIZONS = [30, 60, 90, 120]
# DEFAULT_HORIZONS = [30, 60, 120]
# DEFAULT_HORIZONS = [3, 6, 9]
# DEFAULT_HORIZONS = [2, 3, 4]
DEFAULT_N_JOBS = -1
# DEFAULT_MAX_GENERATIONS = 1
DEFAULT_MAX_GENERATIONS = 10 # default for quick test runs
# DEFAULT_MAX_GENERATIONS = 30
# DEFAULT_MAX_GENERATIONS = 50

# DEFAULT_MODEL_LIST = 'all'
# DEFAULT_MODEL_LIST = 'default'
# DEFAULT_MODEL_LIST = ['default']
# DEFAULT_MODEL_LIST = ['all']
# DEFAULT_MODEL_LIST = ['WindowRegression']
# DEFAULT_MODEL_LIST = ['ETS']
# DEFAULT_MODEL_LIST = ['Theta']
# DEFAULT_MODEL_LIST = ['Prophet']
# DEFAULT_MODEL_LIST = ['UnobservedComponents']
DEFAULT_MODEL_LIST = ['MultivariateMotif']

DEFAULT_TRANSFORMER_LIST = ['default']
# DEFAULT_TRANSFORMER_LIST = ['auto']
# XXX DEFAULT_TRANSFORMER_LIST = 'default' 
##
## Version_2
##

DEFAULT_ENSEMBLE = None
# DEFAULT_ENSEMBLE = 'default'
# DEFAULT_ENSEMBLE = ['auto', 'simple', 'DistanceWeightedEnsemble', 'weighted_ensemble']
# DEFAULT_ENSEMBLE = ['auto']
# default_ensemble = ['auto','simple','horizontal','weighted', 'horizontal-max', 'horizontal-mean']
# DEFAULT_TRANSFORMER_LIST = 'default'
# DEFAULT_MODEL_LIST = ['default']
# DEFAULT_ENSEMBLE = ['auto', 'simple', 'Distance', 'horizontal', 'weighted']
# DEFAULT_ENSEMBLE = 'all'
# DEFAULT_ENSEMBLE = ['all']

DEFAULT_NUM_VALIDATIONS = 'auto'
# DEFAULT_NUM_VALIDATIONS = 2
# DEFAULT_NUM_VALIDATIONS = 3
# DEFAULT_NUM_VALIDATIONS = 5

# DEFAULT_ENABLE_NUM_VALIDATIONS_BACKOFF ，如果啟用，子腳本將在較長 horizon 上自動減少 num_validations 以節省時間（例如，從 3 減少到 2 或 1）。這對於長 horizon 特別有用，因為它們通常需要更多時間來評估模型。默認為 True。

DEFAULT_ENABLE_NUM_VALIDATIONS_BACKOFF = True

# metric_weighting example: '{"mae_weighting": 5, "smape_weighting": 1, "rmse_weighting": 0}' or '{"MAE": 5, "SMAPE": 1, "RMSE": 0}' or 'mae:5,smape:1,rmse:0'
DEFAULT_METRIC_WEIGHTING = None

# if DEFAULT_METRIC_WEIGHTING is None, the launcher will forward a default that ensures mae has the highest weighting (e.g. mae_weighting=5, smape_weighting=1, rmse_weighting=0) to prioritize mae in AutoTS optimization while still including other metrics with lower weight.
DEFAULT_INPUT_FILE = None

##  If DEFAULT_SEEDS_FILE_NAME is None, the launcher will attempt to autodiscover any CSV file matching 'autots_seeds*.csv' under 'AutoTs_SeedGen' or 'input' directories in the script's parent hierarchy. If DEFAULT_SEEDS_FILE_NAME is set to a specific filename, the launcher will only look for that exact file (relative to script dir and cwd) and abort if not found.
# DEFAULT_LOOP = False
DEFAULT_LOOP = False

# DEFAULT_SEEDS_FILE_NAME = 'autots_seeds_20260420_034832.csv'
DEFAULT_SEEDS_FILE_NAME = "autots_seeds_20260420_034832-1seed.csv"
# DEFAULT_SEEDS_FILE_NAME = "autots_seeds_20260420_034832-5seed.csv"
# DEFAULT_SEEDS_FILE_NAME = 'autots_seeds_20260420_034832-30seed.csv'
# DEFAULT_SEEDS_FILE_NAME = "\\input\\autots_seeds_20260420_034832-1seed.csv"

## DEFAULT_SEEDS_FILE_NAME 的行為說明： 
# DEFAULT_OUTPUT_TAG = None  # optional tag to include in child output path for easier identification (e.g. "test1"); if None, no tag is used
# DEFAULT_OUTPUT_TAG = '6vD2_Lancher-S0_G15-30Seed'
DEFAULT_OUTPUT_TAG = Path(__file__).resolve().stem.replace('6vG3-Lancher-', '')  # auto-generate tag from filename for convenience

DEFAULT_START_SEED_INDEX = 0
# DEFAULT_START_SEED_INDEX = "auto"  # can be 'auto' to auto-resume from latest effective_settings*.json seed, or numeric (None/0/1=first seed, 2=second seed, etc.)

def find_latest_effective_settings(output_root: Path):
    try:
        candidates = list(Path(output_root).rglob('effective_settings*.json'))
    except Exception:
        return None
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def check_effective_settings_file(path: Path, expected_seed=None, expected_forbid=True, expected_on_override='fail'):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            j = json.load(f)
    except Exception as e:
        return False, f'cannot_read_json:{e}'
    if expected_seed is not None:
        try:
            if int(j.get('random_seed')) != int(expected_seed):
                return False, f'seed_mismatch({j.get("random_seed")})'
        except Exception:
            return False, 'seed_mismatch'
    if expected_forbid and j.get('FORBID_MODEL_OVERRIDE') is not True:
        return False, f'FORBID_MODEL_OVERRIDE={j.get("FORBID_MODEL_OVERRIDE")}'
    if j.get('ON_OVERRIDE_ACTION') != expected_on_override:
        return False, f'ON_OVERRIDE_ACTION={j.get("ON_OVERRIDE_ACTION")}'
    return True, 'ok'


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if s in ('0', 'false', 'no', 'n', 'off'):
        return False
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch AutoTS script with custom parameters')
    parser.add_argument('--horizons', nargs='+', type=int, default=DEFAULT_HORIZONS)
    parser.add_argument('--n_jobs', type=int, default=DEFAULT_N_JOBS)
    parser.add_argument('--max_generations', type=int, default=DEFAULT_MAX_GENERATIONS)
    parser.add_argument('--num_validations', type=str, default=DEFAULT_NUM_VALIDATIONS,
                        help="AutoTS num_validations forwarded to child script. Use 'auto' or a positive integer.")
    parser.add_argument('--enable_num_validations_backoff', type=str, default=str(DEFAULT_ENABLE_NUM_VALIDATIONS_BACKOFF),
                        help='Enable horizon-level num_validations backoff in child script. true/false')
    parser.add_argument('--transformer_list', nargs='+', default=DEFAULT_TRANSFORMER_LIST)
    parser.add_argument('--model_list', nargs='+', default=DEFAULT_MODEL_LIST)
    parser.add_argument('--ensemble', default=DEFAULT_ENSEMBLE)
    parser.add_argument('--input_file', default=DEFAULT_INPUT_FILE)
    parser.add_argument('--seeds_file', default=None, help='CSV file with seeds (autodiscover AutoTs_SeedGen)')
    parser.add_argument('--loop', action='store_true', default=DEFAULT_LOOP, help='Repeat be tess until interrupted')
    parser.add_argument('--output_tag', default=DEFAULT_OUTPUT_TAG, help='Optional tag to include in child output path')
    parser.add_argument('--output_dir', default='.', help="Child output dir (relative to script dir) or absolute path; default='.'")
    parser.add_argument('--metric_weighting', type=str, default=DEFAULT_METRIC_WEIGHTING,
                        help='JSON string or path to JSON file for AutoTS metric_weighting. If omitted launcher forwards default ensuring mae highest.')
    parser.add_argument('--seeds_start_index', type=str, default=str(DEFAULT_START_SEED_INDEX),
                        help="Start from this index in seeds list. Use 'auto' to resume from latest effective_settings*.json random_seed, or numeric value (None/0/1=first seed, 2=second seed, etc.)")
    args = parser.parse_args()

    # Normalize num_validations: allow only 'auto' or positive integers.
    try:
        raw_nv = str(args.num_validations).strip()
        if raw_nv.lower() == 'auto':
            args.num_validations = 'auto'
        else:
            nv = int(raw_nv)
            if nv < 1:
                raise ValueError('must be >= 1')
            args.num_validations = str(nv)
    except Exception:
        parser.error("--num_validations must be 'auto' or a positive integer")

    # Normalize boolean launcher switch to explicit True/False string for child CLI.
    parsed_backoff = _parse_bool(args.enable_num_validations_backoff)
    if parsed_backoff is None:
        parser.error("--enable_num_validations_backoff must be true or false")
    args.enable_num_validations_backoff = 'True' if parsed_backoff else 'False'

    # Sanitize model_list: remove JSON/dict entries to avoid child converting them to initial_template
    try:
        clean_models = []
        for m in args.model_list:
            if isinstance(m, str):
                try:
                    parsed = json.loads(m)
                    if isinstance(parsed, (dict, list)):
                        print('Launcher: filtered out JSON model_list entry to avoid child initial_template:', m)
                        continue
                except Exception:
                    pass
                clean_models.append(m)
        if not clean_models:
            clean_models = list(DEFAULT_MODEL_LIST)
        args.model_list = clean_models
    except Exception:
        # fallback to defaults if sanitization fails
        args.model_list = list(DEFAULT_MODEL_LIST)

    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '6vG1_autoTs_WeatherToDayWh.py'))

    def find_venv_python(start_path: Path):
        for p in [start_path, *start_path.parents]:
            v = p / '.venv'
            if v.exists():
                if os.name == 'nt':
                    candidate = v / 'Scripts' / 'python.exe'
                else:
                    candidate = v / 'bin' / 'python'
                if candidate.exists():
                    return str(candidate)
        return None

    def find_seeds_file(start_path: Path, filename: str):
        """Locate a seeds file.

        - If filename is None: autodiscover the first matching `autots_seeds*.csv`
          under `AutoTs_SeedGen` then `input` walking up parent directories.
        - If filename looks like a path (contains separators or is absolute): try
          that path (resolve relative to start_path and cwd).
        - Otherwise treat filename as a basename and search `AutoTs_SeedGen` and
          `input` under start_path and its parents.
        Returns a string path or None.
        """
        parents = [start_path, *start_path.parents]

        # autodiscover
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

        # filename provided: if it looks like a path, try as-is or relative to script dir/cwd
        try:
            fpath = Path(filename)
        except Exception:
            fpath = None
        if fpath and (fpath.is_absolute() or (os.sep in filename) or ('/' in filename) or ('\\' in filename)):
            cand = Path(filename)
            if not cand.is_absolute():
                cand = start_path / filename
            if cand.exists():
                return str(cand)
            cand2 = Path.cwd() / filename
            if cand2.exists():
                return str(cand2)
            return None

        # treat as plain filename and search candidate dirs
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
                # try DictReader first
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
                # fallback: simple CSV first column
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

    def _normalize_seeds_start_index(value, total_seeds):
        """Normalize seeds_start_index parameter.
        
        None/0/1 -> 0 (first seed)
        n >= 2 -> n - 1 (0-based index)
        Returns normalized 0-based index, or None if out of range.
        """
        if value is None or value == 0 or value == 1:
            normalized = 0
        elif value >= 2:
            normalized = value - 1
        else:
            return None
        
        if normalized >= total_seeds:
            return None
        return normalized

    python_exe = find_venv_python(Path(__file__).resolve().parent) or sys.executable

    base_cmd = [
        python_exe,
        script_path,
        '--horizons', *map(str, args.horizons),
        '--n_jobs', str(args.n_jobs),
        '--max_generations', str(args.max_generations),
        '--num_validations', str(args.num_validations),
        '--enable_num_validations_backoff', str(args.enable_num_validations_backoff),
        '--transformer_list', *args.transformer_list,
        # force child to refuse model override; on override, child will fail instead of silently falling back
        '--forbid_model_override', 'True',
        '--on_override_action', 'fail',
        '--allow_transformer_retry', 'False',
        '--model_list', *args.model_list,
    ]

    # Prepare metric_weighting robustly: accept JSON string, path to JSON, or simple k:v list-like text.
    try:
        raw_mw = getattr(args, 'metric_weighting', None)
        mw = None
        if raw_mw:
            # if it's a path to an existing file, load it
            try:
                p = Path(str(raw_mw))
                if p.exists():
                    with open(p, 'r', encoding='utf-8') as mf:
                        mw = json.load(mf)
            except Exception:
                mw = None

        if mw is None and raw_mw:
            s = str(raw_mw).strip()
            # try strict JSON parse
            try:
                mw = json.loads(s)
            except Exception:
                # try replacing single quotes with double quotes
                try:
                    mw = json.loads(s.replace("'", '"'))
                except Exception:
                    # try wrapping as object if looks like key:val list
                    try:
                        if ':' in s and not (s.startswith('{') and s.endswith('}')):
                            s2 = '{' + s.strip().strip('{} ') + '}'
                            s2 = s2.replace("'", '"')
                            mw = json.loads(s2)
                    except Exception:
                        mw = None

        if not mw:
            mw = {'mae_weighting': 5, 'smape_weighting': 1, 'rmse_weighting': 0}

        def _normalize_mw(d):
            nd = {}
            def _to_num(x):
                try:
                    if isinstance(x, (int, float)) and not isinstance(x, bool):
                        return float(x)
                    if isinstance(x, str):
                        s = x.strip().strip('"\'"').strip()
                        return float(s)
                except Exception:
                    pass
                return x

            for k, v in (d.items() if isinstance(d, dict) else []):
                kk = str(k).strip()
                vv = _to_num(v)
                if kk.lower().endswith('_weighting'):
                    nd[kk.lower()] = vv
                else:
                    ku = kk.upper()
                    if ku == 'MAE':
                        nd['mae_weighting'] = vv
                    elif ku == 'SMAPE':
                        nd['smape_weighting'] = vv
                    elif ku == 'RMSE':
                        nd['rmse_weighting'] = vv
                    elif ku == 'MASE':
                        nd['mase_weighting'] = vv
                    else:
                        nd[kk.lower()] = vv

            try:
                others = [float(x) for kk, x in nd.items() if kk != 'mae_weighting' and isinstance(x, (int, float))]
                max_other = max(others) if others else 0.0
            except Exception:
                max_other = 0.0
            nd['mae_weighting'] = max(float(nd.get('mae_weighting', 0)), max_other + 1)
            return nd

        mw_norm = _normalize_mw(mw)
        # Pass normalized metric weighting as JSON string directly to child process
        base_cmd += ['--metric_weighting', json.dumps(mw_norm, ensure_ascii=False)]
    except Exception:
        pass

    # determine seeds file (strict rules):
    script_dir = Path(__file__).resolve().parent
    # Determine child output root. Default per user requirement:
    # <script_dir>/output/default/<DEFAULT_OUTPUT_TAG>
    try:
        od = str(args.output_dir).strip()
    except Exception:
        od = '.'
    if od == '' or od == '.' or od == './':
        child_output_root = script_dir / 'output' / 'default' / DEFAULT_OUTPUT_TAG
    else:
        cand = Path(od)
        if not cand.is_absolute():
            child_output_root = (script_dir / cand).resolve()
        else:
            child_output_root = cand
    print(f'Launcher: using child_output_root={child_output_root}')
    seeds_file = None
    seeds_source = None  # 'user', 'default', 'autodiscover', None

    # 1) --seeds_file has highest priority and is strict
    if args.seeds_file:
        found = find_seeds_file(script_dir, args.seeds_file)
        if found:
            seeds_file = found
            seeds_source = 'user'
        else:
            # try literal resolution relative to script dir or cwd
            p = Path(args.seeds_file)
            if not p.is_absolute():
                p = script_dir / args.seeds_file
            if p.exists():
                seeds_file = str(p)
                seeds_source = 'user'
            else:
                print(f"Launcher: specified --seeds_file '{args.seeds_file}' not found; aborting.")
                sys.exit(1)
    else:
        # no explicit file; behavior depends on DEFAULT_SEEDS_FILE_NAME
        if DEFAULT_SEEDS_FILE_NAME is None:
            # autodiscover any autots_seeds*.csv in AutoTs_SeedGen or input
            found = find_seeds_file(script_dir, None)
            if found:
                seeds_file = found
                seeds_source = 'autodiscover'
            else:
                seeds_file = None
                seeds_source = None
        else:
            # only attempt the configured default filename; if not found -> abort
            found = find_seeds_file(script_dir, DEFAULT_SEEDS_FILE_NAME)
            if found:
                seeds_file = found
                seeds_source = 'default'
            else:
                print(f"Launcher: default seeds file '{DEFAULT_SEEDS_FILE_NAME}' not found; aborting.")
                sys.exit(1)

    seeds = []
    if seeds_file:
        seeds = load_seeds(seeds_file)
        if not seeds:
            if seeds_source in ('user', 'default'):
                print(f"Launcher: seeds file '{seeds_file}' found but contains no valid seeds; aborting.")
                sys.exit(1)
            else:
                # autodiscover returned a file that produced no seeds; fallback to single run
                print(f"Launcher: autodiscovered seeds file '{seeds_file}' contained no valid seeds; falling back to single run.")
                seeds = []

    if seeds:
        total = len(seeds)
        # Resolve seeds_start_index (supports both numeric and 'auto' modes)
        start_index = _resolve_seeds_start_index(args.seeds_start_index, seeds, child_output_root)
        
        if start_index is None:
            start_index = 0
        
        if start_index > 0:
            print(f'Launcher: starting from seed index {start_index} (seed #{start_index + 1}/{total})')
        
        for enum_index, s in enumerate(seeds[start_index:], start=start_index):
            actual_position = enum_index + 1  # 1-based display position
            cmd = list(base_cmd)
            cmd += ['--random_seed', str(s)]
            if args.ensemble is not None:
                cmd += ['--ensemble', str(args.ensemble)]
            if args.input_file:
                cmd += ['--input_file', args.input_file]
            # preserve explicit user loop flag to child if set
            if args.loop:
                cmd += ['--loop']
            # forward output tag and output root (script dir) to child
            if args.output_tag:
                cmd += ['--output_tag', args.output_tag]
            # 傳入相對路徑 '.' 給 child，讓 child 在自己的 'output/' 根目錄下建立輸出
            cmd += ['--output_dir', '.']

            print(f'Running seed {actual_position}/{total}:', ' '.join(map(str, cmd)))
            try:
                subprocess.run(cmd, check=True)
                # Post-run detection: verify effective_settings.json exists and enforces forbid override
                out_root = child_output_root
                found = None
                timeout_sec = 30
                interval = 1
                for _poll in range(int(timeout_sec / interval)):
                    p = find_latest_effective_settings(out_root)
                    if p:
                        found = p
                        break
                    time.sleep(interval)
                logp = os.path.join(os.path.dirname(__file__), 'launcher_errors.log')
                if not found:
                    try:
                        with open(logp, 'a', encoding='utf-8') as lf:
                            lf.write(f"{datetime.now().isoformat()} - seed {actual_position}/{total} - post-run check failed: effective_settings.json not found\n")
                    except Exception:
                        pass
                    print(f'Post-run check failed: effective_settings.json not found; logged to {logp}')
                else:
                    ok, msg = check_effective_settings_file(found, expected_seed=int(s))
                    if not ok:
                        try:
                            with open(logp, 'a', encoding='utf-8') as lf:
                                lf.write(f"{datetime.now().isoformat()} - seed {actual_position}/{total} - post-run check failed: {msg} - file: {found}\n")
                        except Exception:
                            pass
                        print(f'Post-run check failed: {msg}; logged to {logp}')
                    else:
                        print('Post-run check passed.')
            except subprocess.CalledProcessError as e:
                logp = os.path.join(os.path.dirname(__file__), 'launcher_errors.log')
                try:
                    with open(logp, 'a', encoding='utf-8') as lf:
                        lf.write(f"{datetime.now().isoformat()} - seed {actual_position}/{total} - cmd: {' '.join(map(str, cmd))} - exit: {getattr(e, 'returncode', 'unknown')}\n")
                        lf.write(str(e) + '\n')
                except Exception:
                    pass
                print(f'Child process failed for seed {actual_position}/{total}; logged to {logp} and continuing.')
                continue
    else:
        # fallback: single run (no seeds)
        cmd = list(base_cmd)
        if args.loop:
            cmd += ['--loop']
        if args.ensemble is not None:
            cmd += ['--ensemble', str(args.ensemble)]
        if args.input_file:
            cmd += ['--input_file', args.input_file]
        # forward output tag and output root (script dir) to child
        if args.output_tag:
            cmd += ['--output_tag', args.output_tag]
        # 傳入相對路徑 '.' 給 child，讓 child 在自己的 'output/' 根目錄下建立輸出
        cmd += ['--output_dir', '.']

        print('Running:', ' '.join(map(str, cmd)))
        try:
            
            subprocess.run(cmd, check=True)
            # Post-run detection: verify effective_settings.json exists and enforces forbid override
            out_root = child_output_root
            found = None
            timeout_sec = 30
            interval = 1
            for _poll in range(int(timeout_sec / interval)):
                p = find_latest_effective_settings(out_root)
                if p:
                    found = p
                    break
                time.sleep(interval)
            logp = os.path.join(os.path.dirname(__file__), 'launcher_errors.log')
            if not found:
                try:
                    with open(logp, 'a', encoding='utf-8') as lf:
                        lf.write(f"{datetime.now().isoformat()} - single run - post-run check failed: effective_settings.json not found\n")
                except Exception:
                    pass
                print(f'Post-run check failed: effective_settings.json not found; logged to {logp}')
            else:
                ok, msg = check_effective_settings_file(found)
                if not ok:
                    try:
                        with open(logp, 'a', encoding='utf-8') as lf:
                            lf.write(f"{datetime.now().isoformat()} - single run - post-run check failed: {msg} - file: {found}\n")
                    except Exception:
                        pass
                    print(f'Post-run check failed: {msg}; logged to {logp}')
                else:
                    print('Post-run check passed.')

        except subprocess.CalledProcessError as e:
            logp = os.path.join(os.path.dirname(__file__), 'launcher_errors.log')
            try:
                with open(logp, 'a', encoding='utf-8') as lf:
                    lf.write(f"{datetime.now().isoformat()} - single run - cmd: {' '.join(map(str, cmd))} - exit: {getattr(e, 'returncode', 'unknown')}\n")
                    lf.write(str(e) + '\n')
            except Exception:
                pass
            print(f'Child process failed for single run; logged to {logp}. Exiting.')
            sys.exit(getattr(e, 'returncode', 1))



