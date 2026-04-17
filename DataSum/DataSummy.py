"""DataSummy — per-file statistical summaries for CSVs in the module's input/ directory.

This script iterates DataSum/input/*.csv (or recursive), computes per-column statistics
and (optionally) generates plots. Outputs are written to DataSum/output/ by default.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("DataSummy")

# ensure local utils import works when running the script directly
pkg_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(pkg_dir))
import utils


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="DataSummy — per-file CSV summaries (for thesis figures)")
	p.add_argument("--input-dir", "-i", default=str(Path(__file__).resolve().parent / "input"), help="input directory containing CSV files")
	p.add_argument("--output-dir", "-o", default=str(Path(__file__).resolve().parent / "output"), help="output directory for summaries and plots")
	p.add_argument("--date-column", default=None, help="name of datetime column (try auto-detect if omitted)")
	p.add_argument("--value-columns", default=None, help="comma-separated numeric columns to summarise (default=all numeric)")
	p.add_argument("--no-charts", dest="charts", action="store_false", help="disable chart generation")
	p.add_argument("--chart-style", choices=["basic", "advanced"], default="advanced", help="chart style when charts enabled")
	p.add_argument("--encoding", default=None, help="file encoding to try first (defaults to utf-8 then latin-1)")
	p.add_argument("--recursive", action="store_true", help="recursively find CSVs under input-dir")
	p.add_argument("--xlsx", action="store_true", help="also write an Excel workbook (requires openpyxl)")
	p.set_defaults(charts=True)
	return p.parse_args(argv)


def _collect_paths(input_dir: Path, recursive: bool) -> List[Path]:
	if recursive:
		return sorted(input_dir.rglob("*.csv"))
	return sorted(input_dir.glob("*.csv"))


def _to_list(opt: Optional[str]) -> Optional[List[str]]:
	if not opt:
		return None
	return [s.strip() for s in opt.split(",") if s.strip()]


def main(argv: Optional[List[str]] = None) -> int:
	args = parse_args(argv)
	input_dir = Path(args.input_dir)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	if not input_dir.exists():
		logger.error("input directory not found: %s", input_dir)
		return 2

	paths = _collect_paths(input_dir, args.recursive)
	if not paths:
		logger.error("No CSV files found in %s", input_dir)
		return 2

	value_columns = _to_list(args.value_columns)

	for p in paths:
		logger.info("Processing %s", p)
		try:
			df = utils.robust_read_csv(str(p), date_column=args.date_column, encoding=args.encoding)
		except Exception:
			logger.exception("Failed to read %s, skipping", p)
			continue
		if df.empty:
			logger.warning("%s is empty, skipping", p)
			continue

		# determine date column
		date_col = args.date_column if args.date_column in df.columns else None
		if not date_col:
			date_col = utils.detect_datetime_col(df)
		if args.date_column and args.date_column not in df.columns:
			logger.warning("Requested date column '%s' not found in %s; auto-detect -> %s", args.date_column, p.name, date_col)

		# numeric & categorical columns
		if value_columns:
			numeric_cols = [c for c in value_columns if c in df.columns]
			missing = set(value_columns) - set(numeric_cols)
			if missing:
				logger.warning("Requested value-columns not found in %s: %s", p.name, ",".join(missing))
		else:
			numeric_cols = df.select_dtypes(include="number").columns.tolist()

		cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns.tolist() if c not in numeric_cols]

		num_summary = utils.compute_numeric_stats(df, numeric_cols) if numeric_cols else pd.DataFrame()
		cat_summary = utils.compute_categorical_stats(df, cat_cols) if cat_cols else pd.DataFrame()

		combined = pd.concat([num_summary, cat_summary], axis=1, sort=False)

		# attach per-file/time metadata
		meta = utils.infer_time_info(df, date_col)
		combined["file"] = p.name
		combined["date_column"] = meta.get("date_column")
		combined["start"] = meta.get("start")
		combined["end"] = meta.get("end")
		combined["inferred_freq"] = meta.get("inferred_freq")

		out_csv = output_dir / f"{p.stem}_summary.csv"
		utils.save_summary(combined, str(out_csv))
		logger.info("Wrote summary %s", out_csv)

		if args.charts and not num_summary.empty:
			plots_dir = output_dir / "plots" / p.stem
			for col in num_summary.index.tolist():
				try:
					utils.make_plots(df, date_col, col, str(plots_dir), style=args.chart_style)
				except Exception:
					logger.exception("Failed to plot %s for %s", col, p.name)

		if args.xlsx:
			try:
				xls_path = output_dir / f"{p.stem}_summary.xlsx"
				with pd.ExcelWriter(xls_path, engine="openpyxl") as w:
					combined.to_excel(w, sheet_name="summary")
				logger.info("Wrote Excel %s", xls_path)
			except Exception:
				logger.exception("Failed to write Excel for %s (openpyxl required)", p.name)

	return 0


if __name__ == "__main__":
	sys.exit(main())


