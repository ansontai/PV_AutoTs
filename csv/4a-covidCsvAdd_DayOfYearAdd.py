from pathlib import Path

import pandas as pd


def main() -> None:
	base = Path(__file__).resolve().parent
	input_path = base / "2000--202602-d-forWh-NoDOY.csv"
	output_path = base / "2000--202602-d-forWh.csv"

	if not input_path.exists():
		raise FileNotFoundError(f"Input file not found: {input_path}")

	df = pd.read_csv(input_path)

	if "Date" in df.columns:
		date_col = "Date"
	elif "LocalTime" in df.columns:
		date_col = "LocalTime"
	else:
		raise ValueError("No Date/LocalTime column found in input CSV")

	dt = pd.to_datetime(df[date_col], errors="coerce")
	df["day_of_year"] = dt.dt.dayofyear

	df.to_csv(output_path, index=False)
	print(f"Written: {output_path}")
	print(f"Rows: {len(df)}")
	print(f"day_of_year nulls: {int(df['day_of_year'].isna().sum())}")


if __name__ == "__main__":
	main()

