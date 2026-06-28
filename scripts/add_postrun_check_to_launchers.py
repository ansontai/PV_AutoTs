#!/usr/bin/env python3
"""Add post-run effective_settings.json checks to launcher scripts.

This script finds files matching a glob pattern under a folder and inserts:
- `import time` near imports if missing
- helper functions `find_latest_effective_settings` and `check_effective_settings_file` if missing
- a post-run detection block after `subprocess.run(cmd, check=True)` occurrences

It creates backups with `.orig` suffix before modifying files.
"""
import argparse
from pathlib import Path
import re
import shutil

POSTRUN_MARKER = 'Post-run detection: verify effective_settings.json'

postrun_helpers = '''
import time
import json

def find_latest_effective_settings(output_root: Path):
    try:
        candidates = list(Path(output_root).rglob('effective_settings.json'))
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
'''


postrun_block_seed = '''
{sub_indent}subprocess.run(cmd, check=True)
{sub_indent}# Post-run detection: verify effective_settings.json exists and enforces forbid override
{sub_indent}out_root = Path(os.path.join(os.path.dirname(__file__), 'output'))
{sub_indent}found = None
{sub_indent}timeout_sec = 30
{sub_indent}interval = 1
{sub_indent}for _poll in range(int(timeout_sec / interval)):
{sub_indent}    p = find_latest_effective_settings(out_root)
{sub_indent}    if p:
{sub_indent}        found = p
{sub_indent}        break
{sub_indent}    time.sleep(interval)
{sub_indent}logp = os.path.join(os.path.dirname(__file__), 'launcher_errors.log')
{sub_indent}if not found:
{sub_indent}    try:
{sub_indent}        with open(logp, 'a', encoding='utf-8') as lf:
{sub_indent}            lf.write(f"{datetime.now().isoformat()} - seed {i}/{total} - post-run check failed: effective_settings.json not found\n")
{sub_indent}    except Exception:
{sub_indent}        pass
{sub_indent}    print(f'Post-run check failed: effective_settings.json not found; logged to {logp}')
{sub_indent}else:
{sub_indent}    ok, msg = check_effective_settings_file(found, expected_seed=int(s))
{sub_indent}    if not ok:
{sub_indent}        try:
{sub_indent}            with open(logp, 'a', encoding='utf-8') as lf:
{sub_indent}                lf.write(f"{datetime.now().isoformat()} - seed {i}/{total} - post-run check failed: {msg} - file: {found}\n")
{sub_indent}        except Exception:
{sub_indent}            pass
{sub_indent}        print(f'Post-run check failed: {msg}; logged to {logp}')
{sub_indent}    else:
{sub_indent}        print('Post-run check passed.')
'''

postrun_block_single = '''
{sub_indent}subprocess.run(cmd, check=True)
{sub_indent}# Post-run detection: verify effective_settings.json exists and enforces forbid override
{sub_indent}out_root = Path(os.path.join(os.path.dirname(__file__), 'output'))
{sub_indent}found = None
{sub_indent}timeout_sec = 30
{sub_indent}interval = 1
{sub_indent}for _poll in range(int(timeout_sec / interval)):
{sub_indent}    p = find_latest_effective_settings(out_root)
{sub_indent}    if p:
{sub_indent}        found = p
{sub_indent}        break
{sub_indent}    time.sleep(interval)
{sub_indent}logp = os.path.join(os.path.dirname(__file__), 'launcher_errors.log')
{sub_indent}if not found:
{sub_indent}    try:
{sub_indent}        with open(logp, 'a', encoding='utf-8') as lf:
{sub_indent}            lf.write(f"{datetime.now().isoformat()} - single run - post-run check failed: effective_settings.json not found\n")
{sub_indent}    except Exception:
{sub_indent}        pass
{sub_indent}    print(f'Post-run check failed: effective_settings.json not found; logged to {logp}')
{sub_indent}else:
{sub_indent}    ok, msg = check_effective_settings_file(found)
{sub_indent}    if not ok:
{sub_indent}        try:
{sub_indent}            with open(logp, 'a', encoding='utf-8') as lf:
{sub_indent}                lf.write(f"{datetime.now().isoformat()} - single run - post-run check failed: {msg} - file: {found}\n")
{sub_indent}        except Exception:
{sub_indent}            pass
{sub_indent}        print(f'Post-run check failed: {msg}; logged to {logp}')
{sub_indent}    else:
{sub_indent}        print('Post-run check passed.')
'''


def process_file(path: Path, dry_run: bool=False):
    text = path.read_text(encoding='utf-8')
    if POSTRUN_MARKER in text:
        return False, 'already_has_postrun'
    modified = False
    # Ensure import time and helpers exist near top (after datetime import if present)
    if "from datetime import datetime" in text and 'def find_latest_effective_settings' not in text:
        text = text.replace('from datetime import datetime', 'from datetime import datetime\n' + postrun_helpers)
        modified = True
    elif 'def find_latest_effective_settings' not in text:
        # try to inject after imports (first occurrence of two newlines after imports)
        m = re.search(r'(import .*\n(?:import .*\n)*)', text)
        if m:
            insert_at = m.end(1)
            text = text[:insert_at] + '\n' + postrun_helpers + text[insert_at:]
            modified = True

    # Insert postrun blocks after subprocess.run occurrences
    # For seeds loop: look for pattern 'for i, s in enumerate(seeds' and a subsequent subprocess.run(cmd, check=True)
    if 'for i, s in enumerate(seeds' in text and 'post-run check failed' not in text:
        # find the subprocess.run call after the seeds loop
        pattern = re.compile(r"(for i, s in enumerate\(seeds[\s\S]*?subprocess.run\(cmd, check=True\))", re.MULTILINE)
        def repl_seed(m):
            run_line = m.group(0)
            # capture indentation from 'subprocess.run' line
            lines = run_line.splitlines()
            for ln in lines[::-1]:
                if 'subprocess.run' in ln:
                    leading = re.match(r"^(\s*)", ln).group(1)
                    break
                block = postrun_block_seed.replace('{sub_indent}', leading)
                return run_line.replace('subprocess.run(cmd, check=True)', block)
        new_text, nsub = pattern.subn(repl_seed, text, count=1)
        if nsub > 0:
            text = new_text
            modified = True

    # For single-run fallback: insert after subprocess.run(cmd, check=True) in single-run branch
    # We look for the 'fallback: single run' comment as anchor
    if 'fallback: single run' in text and 'post-run check failed: effective_settings.json not found' not in text:
        pattern2 = re.compile(r"(fallback: single run \(no seeds\)[\s\S]*?subprocess.run\(cmd, check=True\))", re.MULTILINE)
        def repl_single(m):
            run_line = m.group(0)
            # find indentation
            lines = run_line.splitlines()
            for ln in lines[::-1]:
                if 'subprocess.run' in ln:
                    leading = re.match(r"^(\s*)", ln).group(1)
                    break
            block = postrun_block_single.replace('{sub_indent}', leading)
            return run_line.replace('subprocess.run(cmd, check=True)', block)
        new_text2, nsub2 = pattern2.subn(repl_single, text, count=1)
        if nsub2 > 0:
            text = new_text2
            modified = True

    if modified:
        backup = path.with_suffix(path.suffix + '.orig')
        if not dry_run:
            if not backup.exists():
                shutil.copy(path, backup)
            path.write_text(text, encoding='utf-8', newline='\n')
        return True, 'modified'
    return False, 'no_change'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    parser.add_argument('--pattern', default='*Lancher*.py')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    base = Path(args.folder)
    files = list(base.glob(args.pattern))
    print(f"Scanning {len(files)} files in {base} pattern {args.pattern}")
    report = []
    for f in files:
        ok, reason = process_file(f, dry_run=args.dry_run)
        report.append((str(f), ok, reason))
    for r in report:
        print(r)

if __name__ == '__main__':
    main()
