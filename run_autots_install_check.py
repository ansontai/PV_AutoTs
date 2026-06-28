#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import json
import importlib
import os
import sys
from pathlib import Path


# Prefer project .venv python if present, else fall back to current interpreter
ROOT = Path(__file__).resolve().parent
venv_dir = ROOT / '.venv'
if os.name == 'nt':
    venv_python = venv_dir / 'Scripts' / 'python.exe'
else:
    venv_python = venv_dir / 'bin' / 'python'

python_exe = str(venv_python) if venv_python.exists() else sys.executable


def run_cmd(cmd_list, display_cmd):
    try:
        proc = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out = proc.stdout or ''
        err = proc.stderr or ''
        rc = proc.returncode
    except Exception as e:
        out = ''
        err = str(e)
        rc = 127
    return {'command': display_cmd, 'stdout': out, 'stderr': err, 'exit_code': rc}


def main():
    results = []

    # 1) python -V
    results.append(run_cmd([python_exe, '-V'], f'{python_exe} -V'))

    # 2) python -m pip --version
    results.append(run_cmd([python_exe, '-m', 'pip', '--version'], f'{python_exe} -m pip --version'))

    # 3) Check if AutoTS is installed; install only if missing
    r = run_cmd([python_exe, '-m', 'pip', 'show', 'AutoTS'], f'{python_exe} -m pip show AutoTS || echo "AutoTS not installed"')
    if r['exit_code'] != 0:
        # Auto-install guarded by environment variable `AUTOTS_AUTO_INSTALL`.
        # Set `AUTOTS_AUTO_INSTALL=1` (or 'true','yes') to allow automatic installs.
        env_val = os.environ.get('AUTOTS_AUTO_INSTALL', os.environ.get('AUTOTS_INSTALL', '0'))
        if str(env_val).lower() in ('1', 'true', 'yes', 'y'):
            # not installed -> upgrade pip tools and install AutoTS
            results.append(run_cmd([python_exe, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'], f'{python_exe} -m pip install --upgrade pip setuptools wheel'))
            results.append(run_cmd([python_exe, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade', 'AutoTS'], f'{python_exe} -m pip install --no-cache-dir --upgrade AutoTS'))
            # re-check
            r = run_cmd([python_exe, '-m', 'pip', 'show', 'AutoTS'], f'{python_exe} -m pip show AutoTS || echo "AutoTS not installed after install"')
            if r['exit_code'] != 0:
                r['stdout'] = (r['stdout'] + ("\n" if r['stdout'] else "") + 'AutoTS not installed after install')
        else:
            results.append({'command': 'autots-auto-install-skip', 'stdout': f'AUTOTS_AUTO_INSTALL={env_val!s}; skipping automatic AutoTS install', 'stderr': '', 'exit_code': 0})
    results.append(r)

    # 4) import check for common module names
    buf = ''
    for n in ('autots', 'AutoTS'):
        try:
            m = importlib.import_module(n)
            v = getattr(m, '__version__', getattr(m, 'VERSION', None))
            buf += f'import_ok {n} {v}\n'
        except Exception as e:
            buf += f'import_fail {n} {repr(e)}\n'
    results.append({'command': "import-check (in-process)", 'stdout': buf, 'stderr': '', 'exit_code': 0})

    # Print JSON results
    print(json.dumps(results, ensure_ascii=False))


if __name__ == '__main__':
    main()
