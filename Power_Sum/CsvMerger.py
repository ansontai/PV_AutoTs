import argparse
from pathlib import Path

import pandas as pd


def normalize_date_column(df: pd.DataFrame, candidate_columns: list[str]) -> pd.DataFrame:
	for col in candidate_columns:
		if col in df.columns:
			df = df.copy()
			df["Date"] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
			return df
	raise ValueError(f"No date column found. Tried: {candidate_columns}")


def coalesce_common_suffix_columns(df: pd.DataFrame) -> pd.DataFrame:
	"""Combine pairs like Temperature_a/Temperature_b into one Temperature column."""
	df = df.copy()
	a_suffix = "_a"
	b_suffix = "_b"

	base_names = set()
	for col in df.columns:
		if col.endswith(a_suffix):
			base_names.add(col[: -len(a_suffix)])
		if col.endswith(b_suffix):
			base_names.add(col[: -len(b_suffix)])

	for base in sorted(base_names):
		a_col = f"{base}{a_suffix}"
		b_col = f"{base}{b_suffix}"
		if a_col in df.columns and b_col in df.columns:
			# Prefer historical (a); fallback to forecast (b).
			df[base] = df[a_col].combine_first(df[b_col])
			df = df.drop(columns=[a_col, b_col])

	return df


def merge_csvs(
	file_a: Path,
	file_b: Path,
	output_file: Path,
	how: str = "outer",
	coalesce_common: bool = True,
) -> Path:
	df_a = pd.read_csv(file_a)
	df_b = pd.read_csv(file_b)

	# Support both historical and forecast naming conventions.
	df_a = normalize_date_column(df_a, ["Date", "LocalTime", "date", "local_time", "datetime"])
	df_b = normalize_date_column(df_b, ["Date", "LocalTime", "date", "local_time", "datetime"])

	merged = pd.merge(df_a, df_b, on="Date", how=how, suffixes=("_a", "_b"))
	if coalesce_common:
		merged = coalesce_common_suffix_columns(merged)
	merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
	merged = merged.sort_values("Date")

	# Fill LocalTime for forecast-only rows using Date, if LocalTime exists or is present in original data
	if "LocalTime" in merged.columns:
		# keep format YYYY-MM-DD for LocalTime
		merged["LocalTime"] = merged["LocalTime"].fillna(merged["Date"].dt.strftime("%Y-%m-%d"))
	else:
		# if no LocalTime column, create one from Date
		merged.insert(0, "LocalTime", merged["Date"].dt.strftime("%Y-%m-%d"))

	# Reorder columns: put Date first, then LocalTime (if desired), then the rest
	cols = list(merged.columns)
	# ensure Date exists
	if "Date" in cols:
		cols.remove("Date")
		cols.insert(0, "Date")
	# place LocalTime immediately after Date if present
	if "LocalTime" in cols:
		cols.remove("LocalTime")
		cols.insert(1, "LocalTime")
	merged = merged[cols]

	output_file.parent.mkdir(parents=True, exist_ok=True)
	merged.to_csv(output_file, index=False)
	return output_file


def main() -> None:
	base = Path(__file__).resolve().parent

	parser = argparse.ArgumentParser(description="Merge two CSV files by date.")
	parser.add_argument(
		"--file-a",
		type=Path,
		default=base / "SolarRecord(260204)_d_forWh_WithCodis.csv",
		help="First CSV path (default: SolarRecord file).",
	)
	parser.add_argument(
		"--file-b",
		type=Path,
		default=base / "forecast_weather_1y.csv",
		help="Second CSV path (default: forecast weather file).",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=base / "merged_forecast_solarrecord.csv",
		help="Output CSV path.",
	)
	parser.add_argument(
		"--how",
		choices=["outer", "inner", "left", "right"],
		default="outer",
		help="Join type. Default is outer.",
	)
	parser.add_argument(
		"--no-coalesce",
		action="store_true",
		help="Keep suffix columns (_a/_b) instead of combining common fields.",
	)
	args = parser.parse_args()

	out = merge_csvs(
		args.file_a,
		args.file_b,
		args.output,
		how=args.how,
		coalesce_common=not args.no_coalesce,
	)
	print(f"Merged CSV saved to: {out}")


if __name__ == "__main__":
	main()
