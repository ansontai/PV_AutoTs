#!/usr/bin/env python3
"""
PVGIS_scaler_min-max_forHourly.py

Usage:
  python PVGIS_scaler_min-max_forHourly.py

Reads `csv/SolarRecord_260310_1829-hour-Wh.csv` to get Wh min/max,
reads the PVGIS raw timeseries, aggregates to hourly (mean), then
produces two scaled columns for `P`:
	- P_mapped_Wh         : P mapped to Wh range (map P_min->Wh_min and P_max->Wh_max)
	- P_normalized_0_1_Wh : normalized P in [0,1] using Wh min/max: (P - Wh_min) / (Wh_max - Wh_min)

Default input paths are the ones in the workspace; use command-line
options to override.
"""
from pathlib import Path
from io import StringIO
import argparse
import sys
import pandas as pd


def read_pvgis_raw(fp: Path) -> pd.DataFrame:
	with fp.open("r", encoding="utf-8") as f:
		lines = f.readlines()
	header_idx = None
	for i, line in enumerate(lines):
		if line.strip().lower().startswith("time,"):
			header_idx = i
			break
	if header_idx is None:
		raise RuntimeError(f"Cannot find header line starting with 'time,' in {fp}")
	csv_text = "".join(lines[header_idx:])
	df = pd.read_csv(StringIO(csv_text))
	return df


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--solar", default="csv/SolarRecord_260310_1829-hour-Wh.csv")
	p.add_argument(
		"--pvgis",
		default=(
			"PVGIS/raw/Timeseries_24.148_120.703_E5_0kWp_crystSi_25_35deg_1deg_2005_2005.csv"
		),
	)
	p.add_argument("--out", default=None)
	args = p.parse_args()

	solar_fp = Path(args.solar)
	pvgis_fp = Path(args.pvgis)

	if not solar_fp.exists():
		print(f"Solar CSV not found: {solar_fp}", file=sys.stderr)
		sys.exit(2)
	if not pvgis_fp.exists():
		print(f"PVGIS file not found: {pvgis_fp}", file=sys.stderr)
		sys.exit(2)

	df_solar = pd.read_csv(solar_fp, parse_dates=["LocalTime"], low_memory=False)
	if "Wh" not in df_solar.columns:
		print(f"Solar CSV does not contain 'Wh' column: {solar_fp}", file=sys.stderr)
		sys.exit(2)
	wh_min = float(df_solar["Wh"].min())
	wh_max = float(df_solar["Wh"].max())
	if wh_max == wh_min:
		print("Wh max == Wh min (zero range)", file=sys.stderr)
		sys.exit(2)

	df_pvgis = read_pvgis_raw(pvgis_fp)
	if "time" not in df_pvgis.columns:
		print("PVGIS data missing 'time' column", file=sys.stderr)
		sys.exit(2)
	df_pvgis["time"] = pd.to_datetime(df_pvgis["time"].astype(str), format="%Y%m%d:%H%M", errors="coerce")
	df_pvgis = df_pvgis.dropna(subset=["time"])
	df_pvgis = df_pvgis.set_index("time")

	# convert all columns to numeric where possible before aggregating
	df_pvgis_numeric = df_pvgis.apply(pd.to_numeric, errors="coerce")
	pvgis_hour = df_pvgis_numeric.resample("h").mean()
	pvgis_hour = pvgis_hour.reset_index()

	if "P" not in pvgis_hour.columns:
		print("PVGIS hourly data does not contain 'P' column", file=sys.stderr)
		sys.exit(2)

	pvgis_hour["P"] = pd.to_numeric(pvgis_hour["P"], errors="coerce")
	P_min = float(pvgis_hour["P"].min())
	P_max = float(pvgis_hour["P"].max())
	if P_max == P_min:
		print("P max == P min (zero range)", file=sys.stderr)
		sys.exit(2)

	pvgis_hour["P_mapped_Wh"] = (
		(pvgis_hour["P"] - P_min) / (P_max - P_min) * (wh_max - wh_min) + wh_min
	)
	pvgis_hour["P_normalized_0_1_Wh"] = (pvgis_hour["P"] - wh_min) / (wh_max - wh_min)

	out_fp = (
		Path(args.out)
		if args.out
		else pvgis_fp.with_name(f"{pvgis_fp.stem}[P_scaled_hourly].csv")
	)
	pvgis_hour.to_csv(out_fp, index=False)
	print("Wrote:", out_fp)
	print(f"Wh min/max: {wh_min} / {wh_max}")
	print(f"P (hourly) min/max: {P_min} / {P_max}")


if __name__ == "__main__":
	main()

