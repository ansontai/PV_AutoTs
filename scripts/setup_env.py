#!/usr/bin/env python3
"""Ensure a project-level .venv exists and install requirements only when changed.

Usage:
  python scripts/setup_env.py [--requirements requirements-locked.txt] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
import os


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(4096)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description='Create .venv and install requirements when needed')
    parser.add_argument('--requirements', default=None, help='Requirements file to use (defaults to requirements-locked.txt then requirements.txt)')
    parser.add_argument('--force', action='store_true', help='Force reinstall of requirements')
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    venv_dir = ROOT / '.venv'

    # Select interpreter path inside venv
    if os.name == 'nt':
        venv_python = venv_dir / 'Scripts' / 'python.exe'
    else:
        venv_python = venv_dir / 'bin' / 'python'

    # Choose requirements file
    if args.requirements:
        req_file = Path(args.requirements)
    else:
        if (ROOT / 'requirements-locked.txt').exists():
            req_file = ROOT / 'requirements-locked.txt'
        else:
            req_file = ROOT / 'requirements.txt'

    # Create venv if needed
    if not venv_dir.exists():
        print(f'Creating virtualenv at {venv_dir}')
        subprocess.run([sys.executable, '-m', 'venv', str(venv_dir)], check=True)
    else:
        print(f'.venv exists at {venv_dir}')

    # If no requirements file, nothing to install
    if not req_file.exists():
        print(f'No requirements file found at {req_file}; skipping install')
        return

    # Compute hash and compare with marker
    marker = venv_dir / '.requirements_hash'
    req_hash = file_hash(req_file)
    if args.force or (not marker.exists()) or marker.read_text() != req_hash:
        print('Installing/updating dependencies in .venv...')
        # Upgrade pip tools first
        subprocess.run([str(venv_python), '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'], check=True)
        subprocess.run([str(venv_python), '-m', 'pip', 'install', '--no-cache-dir', '-r', str(req_file)], check=True)
        marker.write_text(req_hash)
        print('Dependencies installed/updated.')
    else:
        print('Requirements unchanged; skipping installation.')


if __name__ == '__main__':
    main()
