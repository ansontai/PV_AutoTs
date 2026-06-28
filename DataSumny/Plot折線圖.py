#!/usr/bin/env python3
"""Plot WH vs date from a SolarRecord CSV file.

This script auto-detects a date-like column and a WH/Wh column, then saves a
line chart with date on the X axis and WH on the Y axis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


DEFAULT_CSV_NAME = "SolarRecord(260228)_d_forWh_WithCodis[date].csv"
DATE_CANDIDATES = ["date", "Date", "LocalTime", "time", "Time", "日期"]
VALUE_CANDIDATES = ["WH", "Wh", "wh"]


def resolve_csv_path(raw_path: str | None) -> Path:
	script_dir = Path(__file__).resolve().parent
	workspace_root = script_dir.parent

	if raw_path:
		candidate = Path(raw_path)
		if candidate.exists():
			return candidate
		if not candidate.is_absolute():
			for base in (script_dir, workspace_root, Path.cwd()):
				resolved = (base / candidate).resolve()
				if resolved.exists():
					return resolved
		raise FileNotFoundError(f"找不到 CSV 檔案: {raw_path}")

	for base in (script_dir, workspace_root, Path.cwd()):
		candidate = (base / DEFAULT_CSV_NAME).resolve()
		if candidate.exists():
			return candidate

	search_roots = [script_dir, workspace_root, Path.cwd()]
	for root in search_roots:
		try:
			for found in root.rglob(DEFAULT_CSV_NAME):
				return found
		except Exception:
			continue

	raise FileNotFoundError(f"找不到預設 CSV 檔案: {DEFAULT_CSV_NAME}")


def detect_column(columns: list[str], candidates: list[str]) -> str | None:
	lower_map = {str(col).strip().lower(): str(col) for col in columns}
	for candidate in candidates:
		key = candidate.strip().lower()
		if key in lower_map:
			return lower_map[key]
	return None


def build_plot(csv_path: Path, output_path: Path, date_column: str | None, value_column: str | None) -> Path:
	df = pd.read_csv(csv_path)

	resolved_date_col = date_column or detect_column(list(df.columns), DATE_CANDIDATES)
	if resolved_date_col is None:
		raise ValueError(f"找不到日期欄位，現有欄位：{list(df.columns)}")

	resolved_value_col = value_column or detect_column(list(df.columns), VALUE_CANDIDATES)
	if resolved_value_col is None:
		raise ValueError(f"找不到 WH 欄位，現有欄位：{list(df.columns)}")

	df[resolved_date_col] = pd.to_datetime(df[resolved_date_col], errors="coerce")
	df[resolved_value_col] = pd.to_numeric(df[resolved_value_col], errors="coerce")
	df = df.dropna(subset=[resolved_date_col, resolved_value_col]).sort_values(resolved_date_col)

	if df.empty:
		raise ValueError("日期欄與 WH 欄轉換後沒有可繪圖的資料")

	fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
	ax.plot(df[resolved_date_col], df[resolved_value_col], color="dimgray", linewidth=1.8)
	ax.set_title(f"{resolved_value_col} 日期折線圖", fontsize=13)
	ax.set_xlabel("日期")
	ax.set_ylabel(resolved_value_col)
	ax.grid(True, alpha=0.3, linestyle=":")

	ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
	ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
	plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

	fig.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	return output_path


def main() -> int:
	parser = argparse.ArgumentParser(description="根據 CSV 的日期欄與 WH 欄繪製折線圖")
	parser.add_argument("csv_path", nargs="?", default=None, help="CSV 檔案路徑；不填則自動尋找預設檔名")
	parser.add_argument("--date-column", default=None, help="手動指定日期欄位名稱")
	parser.add_argument("--value-column", default=None, help="手動指定數值欄位名稱，預設自動找 WH/Wh")
	parser.add_argument("--output", default=None, help="輸出 PNG 路徑；不填則與 CSV 同名")
	args = parser.parse_args()

	csv_path = resolve_csv_path(args.csv_path)
	output_path = Path(args.output) if args.output else csv_path.with_suffix(".png")
	output_path.parent.mkdir(parents=True, exist_ok=True)

	saved_path = build_plot(
		csv_path=csv_path,
		output_path=output_path,
		date_column=args.date_column,
		value_column=args.value_column,
	)
	print(f"已輸出圖檔: {saved_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
