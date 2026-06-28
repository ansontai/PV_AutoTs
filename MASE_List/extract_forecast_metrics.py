#!/usr/bin/env python3
"""
搜尋資料夾內的 metrics JSON 或 effective_settings CSV 檔案，
並把每個檔案的欄位數值轉成 list 後匯出到 CSV。

輸出格式：每列對應一個來源檔，欄位為所有出現過的 keys，前兩欄為 `category`（父目錄名稱）與 `file`（相對於輸入資料夾的路徑或絕對路徑）。
"""
import argparse
import csv
import json
import logging
import re
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import statistics
import math


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


# Keep previous symbol name
_generate_summary = process_file


def find_files(base: Path, recursive: bool, pattern: str) -> List[Path]:
    """Find files matching `pattern` under `base`.

    If `recursive` is True use rglob, otherwise use glob.
    `pattern` is a glob pattern such as 'forecast_Wh_metrics_*d.json'.
    """
    if recursive:
        return list(base.rglob(pattern))
    return list(base.glob(pattern))


def load_json_safe(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception as e:
        logging.warning('Failed to load JSON %s: %s', p, e)
        return None


def load_csv_safe(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with p.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                logging.warning('CSV has no data rows in %s', p)
                return None
            if len(rows) > 1:
                logging.warning('CSV has multiple data rows in %s; using the first row only', p)
            return rows[0]
    except Exception as e:
        logging.warning('Failed to load CSV %s: %s', p, e)
        return None


def load_record(p: Path) -> Optional[Dict[str, Any]]:
    suffix = p.suffix.lower()
    if suffix == '.csv':
        return load_csv_safe(p)
    if suffix == '.json':
        return load_json_safe(p)

    data = load_json_safe(p)
    if data is not None:
        return data
    return load_csv_safe(p)


def normalize_value(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def parse_numeric_value(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        num = float(raw)
        return None if math.isnan(num) else num
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            num = float(s)
        except Exception:
            return None
        return None if math.isnan(num) else num
    return None


def _sanitize_category_name(cat: Any) -> str:
    s = str(cat) if cat is not None else ''
    sanitized = re.sub(r'[^A-Za-z0-9._-]+', '_', s)
    sanitized = sanitized.strip('_')
    return sanitized or 'uncategorized'


def write_per_category(records: List[Dict[str, Any]], columns: List[str], out_path: Path, encoding: str) -> None:
    """Write one CSV per `category` under `out_path.parent / 'by_category'`.

    Filenames are `{out_path.stem}__category-{sanitized}.csv`. If multiple
    different categories produce the same sanitized name, append a short
    sha1 hash to avoid collisions.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        cat = rec.get('category', '')
        groups[str(cat)].append(rec)

    if not groups:
        logging.info('No categories found to split.')
        return

    out_dir = out_path.parent / 'by_category'
    out_dir.mkdir(parents=True, exist_ok=True)

    used_names = set()
    for cat in sorted(groups.keys()):
        items = groups[cat]
        base = _sanitize_category_name(cat)
        name = base
        if name in used_names:
            h = hashlib.sha1(cat.encode('utf-8')).hexdigest()[:8]
            name = f"{base}-{h}"
        used_names.add(name)

        filename = f"{out_path.stem}__category-{name}.csv"
        path = out_dir / filename
        try:
            with path.open('w', newline='', encoding=encoding) as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()
                for rec in items:
                    row: Dict[str, str] = {}
                    for col in columns:
                        if col in ('category', 'file'):
                            row[col] = str(rec.get(col, ''))
                        else:
                            val = rec.get(col, None)
                            norm = normalize_value(val)
                            row[col] = json.dumps(norm, ensure_ascii=False)
                    writer.writerow(row)
            logging.info('Wrote %d rows to %s', len(items), path)
            if _generate_summary is not None:
                try:
                    _generate_summary(path, encoding=encoding)
                except Exception as e:
                    logging.exception('Failed to generate summary for %s: %s', path, e)
        except Exception as e:
            logging.exception('Failed to write per-category CSV %s: %s', path, e)


def build_mase_summary_rows(records: List[Dict[str, Any]], holdout_mase_column: str = 'holdout_mase') -> List[Dict[str, Any]]:
    values: List[float] = []
    for rec in records:
        num = parse_numeric_value(rec.get(holdout_mase_column))
        if num is not None:
            values.append(num)

    if not values:
        return []

    median_mase = statistics.median(values)
    summary_row: Dict[str, Any] = {
        'row_type': 'summary',
        'seed_count': len(values),
        'holdout_mase_median': median_mase,
        'category': '',
        'file': '',
    }
    return [summary_row]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description='Extract forecast_Wh_metrics JSON files into a CSV'
    )
    parser.add_argument('folder', help='Folder path to search')
    parser.add_argument('--pattern', '-p', default='forecast_Wh_metrics_*d.json',
                        help='Glob pattern to search for (default: forecast_Wh_metrics_*d.json)')
    parser.add_argument('--output', '-o', default='forecast_metrics_extracted.csv', help='Output CSV path')
    parser.add_argument('--no-recursive', action='store_true', help='Do not search subfolders')
    parser.add_argument('--absolute-paths', action='store_true', help='Store absolute file paths in CSV')
    parser.add_argument('--encoding', default='utf-8-sig', help='CSV encoding (default utf-8-sig)')
    parser.add_argument('--append-mase-summary', action='store_true', help='Append a summary row with holdout_mase median and seed count')
    parser.add_argument('--holdout-mase-column', default='holdout_mase', help='Column name used for summary median/count calculation')
    args = parser.parse_args(argv)

    base = Path(args.folder)
    if not base.exists() or not base.is_dir():
        logging.error('Folder not found or not a directory: %s', base)
        raise SystemExit(2)

    files = find_files(base, recursive=(not args.no_recursive), pattern=args.pattern)
    if not files:
        logging.info('No files matching %s found under %s', args.pattern, base)
        return

    records: List[Dict[str, Any]] = []
    all_keys = set()

    for f in sorted(files):
        data = load_record(f)
        if data is None:
            continue
        if not isinstance(data, dict):
            logging.warning('JSON root is not an object in %s; skipping', f)
            continue
        all_keys.update(data.keys())
        category = f.parent.name or ''
        file_field = f.resolve().as_posix() if args.absolute_paths else f.relative_to(base).as_posix()
        record: Dict[str, Any] = {'category': category, 'file': file_field}
        for k, v in data.items():
            record[k] = v
        records.append(record)

    columns = ['category', 'file'] + sorted(all_keys)

    if args.append_mase_summary:
        summary_rows = build_mase_summary_rows(records, holdout_mase_column=args.holdout_mase_column)
        if summary_rows:
            records.extend(summary_rows)
            all_keys.update(summary_rows[0].keys())
            columns = ['category', 'file'] + sorted(all_keys)

    out_path = Path(args.output)
    try:
        # Ensure parent directory exists (handles cases like 'output/file.csv')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w', newline='', encoding=args.encoding) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            for rec in records:
                row: Dict[str, str] = {}
                for col in columns:
                    if col in ('category', 'file'):
                        row[col] = str(rec.get(col, ''))
                    else:
                        val = rec.get(col, None)
                        norm = normalize_value(val)
                        row[col] = json.dumps(norm, ensure_ascii=False)
                writer.writerow(row)

        # Also write per-category CSVs under a `by_category/` subfolder.
        write_per_category(records, columns, out_path, args.encoding)
    except Exception as e:
        logging.exception('Failed to write CSV %s: %s', out_path, e)
        raise SystemExit(3)

    logging.info('Wrote %d rows to %s', len(records), out_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
