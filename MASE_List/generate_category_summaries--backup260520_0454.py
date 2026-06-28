#!/usr/bin/env python3
"""
Generate per-category summaries from per-category CSV files.

For each CSV matching `*__category-*.csv` (excluding files that already end with `-summy.csv`), compute:
- records: number of rows
- For each column: values_count (numeric values), min, median, max, mean, std
Write summary CSV named `{original_stem}-summy.csv` in the same folder.
"""
import argparse
import csv
import json
import logging
import statistics
import math
from pathlib import Path
from typing import List, Dict, Any


def parse_cell(raw: str) -> List[Any]:
    if raw is None or raw == '':
        return []
    # Try JSON parse first (per-category CSV stores lists as JSON strings)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except Exception:
        # Not JSON -> try coercing to a single numeric value
        try:
            return [float(raw)]
        except Exception:
            return []


def numeric_values(rows: List[Dict[str, str]], col: str) -> List[float]:
    out: List[float] = []
    for r in rows:
        raw = r.get(col, '')
        vals = parse_cell(raw)
        for v in vals:
            if isinstance(v, bool):
                continue
            try:
                num = float(v)
            except Exception:
                # try cleanup like '1,234' or '12%'
                if isinstance(v, str):
                    s = v.strip().replace(',', '')
                    if s.endswith('%'):
                        try:
                            num = float(s.rstrip('%')) / 100.0
                        except Exception:
                            continue
                    else:
                        try:
                            num = float(s)
                        except Exception:
                            continue
                else:
                    continue
            if math.isnan(num):
                continue
            out.append(num)
    return out


def process_file(path: Path, encoding: str = 'utf-8-sig') -> None:
    with path.open('r', encoding=encoding, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not fieldnames:
        logging.warning('No columns in %s', path)
        return

    summary_rows: List[Dict[str, Any]] = []
    records_count = len(rows)
    for col in fieldnames:
        nums = numeric_values(rows, col)
        values_count = len(nums)
        if values_count:
            try:
                min_v = min(nums)
                max_v = max(nums)
                median_v = statistics.median(nums)
                mean_v = statistics.mean(nums)
                std_v = statistics.pstdev(nums)
            except Exception:
                min_v = median_v = max_v = mean_v = std_v = ''
        else:
            min_v = median_v = max_v = mean_v = std_v = ''

        summary_rows.append({
            'column': col,
            'records': records_count,
            'values_count': values_count,
            'min': min_v,
            'median': median_v,
            'max': max_v,
            'mean': mean_v,
            'std': std_v,
        })

    summary_path = path.with_name(path.stem + '-summy.csv')
    with summary_path.open('w', encoding=encoding, newline='') as sf:
        fields = ['column', 'records', 'values_count', 'min', 'median', 'max', 'mean', 'std']
        w = csv.DictWriter(sf, fieldnames=fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    logging.info('Wrote summary %s', summary_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Generate per-category summaries from per-category CSVs')
    parser.add_argument('folder', help='Folder containing per-category CSVs')
    parser.add_argument('--pattern', default='*__category-*.csv', help='Glob pattern for per-category CSVs')
    parser.add_argument('--encoding', default='utf-8-sig', help='CSV encoding')
    args = parser.parse_args(argv)

    base = Path(args.folder)
    if not base.exists() or not base.is_dir():
        logging.error('Folder not found or not a directory: %s', base)
        raise SystemExit(2)

    files = sorted(base.glob(args.pattern))
    for p in files:
        if p.name.endswith('-summy.csv'):
            continue
        process_file(p, encoding=args.encoding)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
