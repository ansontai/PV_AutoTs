"""
PVGIS TMY daily CSV helper.

Input:
	Timeseries_24.148_120.703_E5_0kWp_crystSi_25_35deg_1deg_2005_2005[UTC+8][daily][scaled].csv

Behavior:
  - Copy the `date` column into a new column named `_date(org)`.
	- Rewrite `date` so the first row starts at a configurable reference date.
  - Save a new CSV with a `[dateAdj]` suffix.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "output"
DEFAULT_INPUT_NAME = "Timeseries_24.148_120.703_E5_0kWp_crystSi_25_35deg_1deg_2005_2005[UTC+8][daily][scaled][vB].csv"
DEFAULT_OUTPUT_SUFFIX = "[dateAdj]"
ORIGINAL_START_DATE = "2005/03/01"
REFERENCE_DATE = "2026/03/01"


def find_input_file(explicit_path: Path | None = None) -> Path:
	if explicit_path is not None:
		if explicit_path.exists():
			return explicit_path
		raise FileNotFoundError(f"找不到輸入檔：{explicit_path}")

	candidates = [
		HERE / "input" / DEFAULT_INPUT_NAME,
		HERE / "input" / "tmy_24.148_120.703_2005_2023[UTC+8][daily][mapped][vB].csv",
		# HERE / "raw" / DEFAULT_INPUT_NAME,
		# HERE / "raw" / "tmy_24.148_120.703_2005_2023[UTC+8][daily].csv",
		# HERE / "my" / DEFAULT_INPUT_NAME,
		# HERE / "my" / "tmy_24.148_120.703_2005_2023[UTC+8][daily].csv",
		# HERE / DEFAULT_INPUT_NAME,
		# HERE / "tmy_24.148_120.703_2005_2023[UTC+8][daily].csv",
	]
	for path in candidates:
		if path.exists():
			return path

	raise FileNotFoundError(
		"找不到輸入檔，已搜尋：\n" + "\n".join(str(path) for path in candidates)
	)


def build_output_path(input_file: Path, suffix: str) -> Path:
	return OUTPUT_DIR / f"{input_file.stem}{suffix}.csv"


def process_csv(input_file: Path, output_file: Path) -> Path:
	df = pd.read_csv(input_file)

	if "date" not in df.columns:
		raise KeyError(f"輸入檔缺少 `date` 欄位：{input_file}")

	original_dates = pd.to_datetime(df["date"], errors="coerce")
	start_date = pd.to_datetime(ORIGINAL_START_DATE)
	if original_dates.isna().all():
		raise ValueError(f"無法解析輸入檔的 `date` 欄位：{input_file}")

	start_matches = original_dates == start_date
	if not start_matches.any():
		raise ValueError(f"找不到起始日期 {ORIGINAL_START_DATE}，無法旋轉列順序：{input_file}")

	start_idx = start_matches[start_matches].index[0]
	if start_idx != 0:
		upper = df.iloc[:start_idx].copy()
		lower = df.iloc[start_idx:].copy()
		df = pd.concat([lower, upper], ignore_index=True)

	df["_date(org)"] = df["date"].copy()
	start_date = pd.to_datetime(REFERENCE_DATE)
	adjusted_dates = pd.date_range(start=start_date, periods=len(df), freq="D")
	df["date"] = adjusted_dates.strftime("%Y/%m/%d")

	cols = ["date", "_date(org)"] + [c for c in df.columns if c not in {"date", "_date(org)"}]
	df = df[cols]

	output_file.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(output_file, index=False, encoding="utf-8")
	return output_file


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Copy the date column to _date(org) and save a new CSV."
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=None,
		help="Input CSV path. Defaults to the standard PVGIS file search.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Output CSV path. Defaults to input stem + [dateAdj].csv.",
	)
	parser.add_argument(
		"--suffix",
		default=DEFAULT_OUTPUT_SUFFIX,
		help="Suffix appended to the input stem when --output is not provided.",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()

	try:
		input_file = find_input_file(args.input)
		output_file = args.output or build_output_path(input_file, args.suffix)
		written = process_csv(input_file, output_file)
	except Exception as exc:
		print(f"失敗：{exc}")
		return 1

	print(f"已讀取：{input_file}")
	print(f"已輸出：{written}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
