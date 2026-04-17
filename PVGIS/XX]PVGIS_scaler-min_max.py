#!/usr/bin/env python3
"""
PVGIS_scaler-min_max.py

用途:
  使用 SolarRecord（參考檔）計算 min/max，再對 PVGIS Timeseries 的指定欄位做 min-max 縮放。

範例:
  python PVGIS_scaler-min_max.py -i "PVGIS/output/Timeseries_....csv" -s "PVGIS/my/SolarRecord_260310_1829-daily-1d.csv" -c P_Wh -o "PVGIS/output/Timeseries_minmax_by_SolarRecord.csv" -p "PVGIS/output/pvgis_stats.json" --clip
"""
from pathlib import Path
import argparse
import json
import sys
import pandas as pd


def detect_unit(series, force):
    if force != "auto":
        return force
    med = series.median()
    return "wh" if med >= 1000 else "kwh"


def compute_stats_from_solar(solar_path, ref_col="Wh", force_unit="auto"):
    df = pd.read_csv(solar_path)
    if ref_col not in df.columns:
        raise ValueError(f"Reference column {ref_col} not found in {solar_path}")
    s = pd.to_numeric(df[ref_col], errors="coerce").dropna()
    if s.empty:
        raise ValueError("Reference column contains no numeric values")
    unit = detect_unit(s, force_unit)
    if unit == "kwh":
        s_wh = s * 1000.0
    else:
        s_wh = s.astype(float)
    min_wh = float(s_wh.min())
    max_wh = float(s_wh.max())
    return {
        "ref_path": str(solar_path),
        "ref_col": ref_col,
        "unit_detected": unit,
        "min_wh": min_wh,
        "max_wh": max_wh,
        "min_kwh": min_wh / 1000.0,
        "max_kwh": max_wh / 1000.0,
        "count": int(len(s_wh)),
    }


def load_stats_from_json(path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def scale_series_minmax(series, minv, maxv):
    s = pd.to_numeric(series, errors="coerce")
    denom = maxv - minv
    if denom == 0 or denom is None:
        return s - minv
    return (s - minv) / denom


def column_unit_guess(colname):
    cl = colname.lower()
    if "kwh" in cl:
        return "kwh"
    return "wh"


def main():
    p = argparse.ArgumentParser(description="Min-max scale PVGIS timeseries using SolarRecord reference")
    p.add_argument("-i", "--input", required=True, help="Timeseries CSV input")
    p.add_argument("-s", "--solar", help="SolarRecord CSV used to compute min/max (use --params to load instead)")
    p.add_argument("-p", "--params", help="Stats JSON path to save (when computing) or load (when scaling)")
    p.add_argument("-c", "--cols", nargs="+", default=["P_Wh"], help="Columns to scale (default: P_Wh). If P_Wh present, related P_* columns are auto-included.")
    p.add_argument("--ref-col", default="Wh", help="Reference column name in solar file (default: Wh)")
    p.add_argument("--force-unit", choices=["auto", "wh", "kwh"], default="auto", help="Force interpretation of solar ref column units")
    p.add_argument("-o", "--out", help="Output CSV path (default: input_stem_minmax.csv)")
    p.add_argument("--clip", action="store_true", help="Clip scaled values to [0,1]")
    p.add_argument("--suffix", default="_minmax_by_solar", help="Suffix for new scaled columns")
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print("Error: input not found:", inp, file=sys.stderr)
        sys.exit(1)

    stats = None
    if args.solar:
        solar_path = Path(args.solar)
        if not solar_path.exists():
            print("Error: solar reference not found:", solar_path, file=sys.stderr)
            sys.exit(1)
        stats = compute_stats_from_solar(solar_path, ref_col=args.ref_col, force_unit=args.force_unit)
        if args.params:
            with open(args.params, "w", encoding="utf8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            print("Saved stats to", args.params)
    elif args.params:
        stats = load_stats_from_json(args.params)
    else:
        print("Error: must provide --solar or --params", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(inp)

    # expand P_Wh to include other P_ columns commonly needed
    cols = list(args.cols)
    if "P_Wh" in cols:
        extra = ["P_kWh", "P_mean", "P_min", "P_max", "P_std", "P_p10", "P_p90"]
        for e in extra:
            if e not in cols and e in df.columns:
                cols.append(e)

    for col in cols:
        if col not in df.columns:
            print(f"Warning: column {col} not in {inp}, skipping", file=sys.stderr)
            continue
        # decide conversion to Wh for uniform scaling
        clow = col.lower()
        # if the column is in kWh, convert to Wh
        if "kwh" in clow:
            energy_wh = pd.to_numeric(df[col], errors="coerce") * 1000.0
            minv = stats.get("min_wh")
            maxv = stats.get("max_wh")
            note = "(converted from kWh -> Wh)"
        # if the column is already Wh (explicit P_Wh), use directly
        elif col == "P_Wh":
            energy_wh = pd.to_numeric(df[col], errors="coerce")
            minv = stats.get("min_wh")
            maxv = stats.get("max_wh")
            note = "(Wh)"
        # otherwise treat as power (W) and convert to Wh using n_obs_P if available
        else:
            n_obs_col = "n_obs_P"
            if n_obs_col in df.columns:
                nobs = pd.to_numeric(df[n_obs_col], errors="coerce").fillna(0)
            else:
                # fallback to 24 hours if n_obs not available
                print(f"Warning: {n_obs_col} not found; assuming 24 for conversion of {col}", file=sys.stderr)
                nobs = pd.Series(24, index=df.index)
            energy_wh = pd.to_numeric(df[col], errors="coerce") * nobs
            minv = stats.get("min_wh")
            maxv = stats.get("max_wh")
            note = f"(converted from W using {n_obs_col} -> Wh)"

        if minv is None or maxv is None:
            print(f"Error: stats missing min/max for scaling, skipping {col}", file=sys.stderr)
            continue

        denom = maxv - minv
        if denom == 0:
            scaled = energy_wh - minv
        else:
            scaled = (energy_wh - minv) / denom

        if args.clip:
            scaled = scaled.clip(0, 1)

        new_col = col + args.suffix
        df[new_col] = scaled
        print(f"Scaled {col} -> {new_col} using min={minv}, max={maxv} {note}")

    out_path = Path(args.out) if args.out else inp.with_name(inp.stem + "_minmax" + inp.suffix)
    df.to_csv(out_path, index=False)
    print("Wrote scaled file to", out_path)


if __name__ == "__main__":
    main()
