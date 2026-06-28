"""stat_summary — 計算單一 CSV 的樣本數 N、特徵數 p、以及每個欄位的缺失率。

Usage:
    python DataSum/stat_summary.py --file path/to.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("stat_summary")

# ensure local utils import works when running the script directly
pkg_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(pkg_dir))
import utils


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute N, p and missing-rate per feature for one CSV")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--file", "-f", help="input CSV file")
    grp.add_argument("--input-dir", "-d", help="input directory containing CSV files")
    p.add_argument("--date-column", default=None, help="name of datetime column (optional)")
    p.add_argument("--value-columns", default=None, help="comma-separated columns to include (default=numeric)")
    p.add_argument("--all", dest="all_cols", action="store_true", help="include all columns (not only numeric)")
    p.add_argument("--output-dir", "-o", default=str(pkg_dir / "output"), help="output directory")
    p.add_argument("--recursive", action="store_true", help="recursively find CSVs under input-dir")
    p.add_argument("--merge", action="store_true", help="merge all file summaries into a single CSV (default: per-file outputs)")
    p.add_argument("--xlsx", action="store_true", help="also write an Excel workbook (requires openpyxl)")
    p.add_argument("--encoding", default=None, help="file encoding to try first")
    p.set_defaults(all_cols=False)
    return p.parse_args(argv)


def _to_list(opt: Optional[str]) -> Optional[List[str]]:
    if not opt:
        return None
    return [s.strip() for s in opt.split(",") if s.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    # directory mode
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _process_file(in_path: Path):
        try:
            df = utils.robust_read_csv(str(in_path), date_column=args.date_column, encoding=args.encoding)
        except Exception:
            logger.exception("Failed to read %s", in_path)
            return pd.DataFrame()

        total_n = len(df)
        if args.value_columns:
            requested = _to_list(args.value_columns) or []
            selected_cols = [c for c in requested if c in df.columns]
            missing = set(requested) - set(selected_cols)
            if missing:
                logger.warning("Requested value-columns not found in %s: %s", in_path.name, ",".join(missing))
        elif args.all_cols:
            selected_cols = df.columns.tolist()
        else:
            selected_cols = df.select_dtypes(include="number").columns.tolist()

        p_count = len(selected_cols)
        if p_count == 0:
            logger.warning("No selected columns for summary in %s", in_path.name)

        rows = []
        for c in selected_cols:
            ser = df[c]
            non_null = int(ser.count())
            missing_count = int(total_n - non_null)
            missing_rate = None if total_n == 0 else float(missing_count) / float(total_n)
            if pd.api.types.is_numeric_dtype(ser):
                dt = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(ser):
                dt = "datetime"
            else:
                dt = "categorical"
            rows.append({
                "file": in_path.name,
                "N": total_n,
                "p": p_count,
                "variable": c,
                "type": dt,
                "total_count": total_n,
                "non_null": non_null,
                "missing_count": missing_count,
                "missing_rate": missing_rate,
            })

        return pd.DataFrame(rows)

    all_results: List[pd.DataFrame] = []

    if args.file:
        df_out = _process_file(Path(args.file))
        if df_out.empty and df_out.shape[0] == 0:
            logger.warning("No results for %s", args.file)
        else:
            out_csv = out_dir / f"{Path(args.file).stem}_missing_summary.csv"
            df_out.to_csv(out_csv, index=False)
            logger.info("Wrote summary %s", out_csv)
            if args.xlsx:
                try:
                    xls_path = out_dir / f"{Path(args.file).stem}_missing_summary.xlsx"
                    with pd.ExcelWriter(xls_path, engine="openpyxl") as w:
                        df_out.to_excel(w, sheet_name="missing", index=False)
                    logger.info("Wrote Excel %s", xls_path)
                except Exception:
                    logger.exception("Failed to write Excel for %s (openpyxl required)", args.file)
        return 0

    # directory mode
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error("input directory not found: %s", input_dir)
        return 2

    if args.recursive:
        paths = sorted(input_dir.rglob("*.csv"))
    else:
        paths = sorted(input_dir.glob("*.csv"))

    if not paths:
        logger.error("No CSV files found in %s", input_dir)
        return 2

    for pth in paths:
        df_out = _process_file(pth)
        if df_out.empty:
            continue
        if args.merge:
            all_results.append(df_out)
        else:
            out_csv = out_dir / f"{pth.stem}_missing_summary.csv"
            df_out.to_csv(out_csv, index=False)
            logger.info("Wrote summary %s", out_csv)
            if args.xlsx:
                try:
                    xls_path = out_dir / f"{pth.stem}_missing_summary.xlsx"
                    with pd.ExcelWriter(xls_path, engine="openpyxl") as w:
                        df_out.to_excel(w, sheet_name="missing", index=False)
                    logger.info("Wrote Excel %s", xls_path)
                except Exception:
                    logger.exception("Failed to write Excel for %s (openpyxl required)", pth.name)

    if args.merge and all_results:
        merged = pd.concat(all_results, axis=0, sort=False)
        out_csv = out_dir / f"merged_missing_summary.csv"
        merged.to_csv(out_csv, index=False)
        logger.info("Wrote merged summary %s", out_csv)
        if args.xlsx:
            try:
                xls_path = out_dir / f"merged_missing_summary.xlsx"
                with pd.ExcelWriter(xls_path, engine="openpyxl") as w:
                    merged.to_excel(w, sheet_name="missing", index=False)
                logger.info("Wrote Excel %s", xls_path)
            except Exception:
                logger.exception("Failed to write merged Excel (openpyxl required)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
