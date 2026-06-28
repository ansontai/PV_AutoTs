#!/usr/bin/env python3
"""
搜尋資料夾內的 `forecast_Wh_metrics_{*}d.json`（或符合 glob pattern 的檔案），
並把每個檔案的欄位數值轉成 list 後匯出到 CSV。

輸出格式：每列對應一個 JSON 檔，欄位為所有出現過的 JSON keys，前兩欄為 `category`（父目錄名稱）與 `file`（相對於輸入資料夾的路徑或絕對路徑）。
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


def normalize_value(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


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
        except Exception as e:
            logging.exception('Failed to write per-category CSV %s: %s', path, e)


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
        data = load_json_safe(f)
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
