#!/usr/bin/env python3
"""
PVGIS_scaler-launcher.py

Launcher for `PVGIS_scaler-min_max.py` using local reference files.

Behaviour:
- reference: __file__/my/SolarRecord_260310_1829-daily-1d.csv
- timeseries: __file__/my/Timeseries_24.148_120.703_E5_0kWp_crystSi_25_35deg_1deg_2005_2005[UTC+8][daily].csv
- output: __file__/output (creates if missing)

This script calls the scaler using the current Python interpreter.
"""
from pathlib import Path
import subprocess
import sys


def main():
    base = Path(__file__).resolve().parent

    solar = base / "my" / "SolarRecord_260310_1829-daily-1d.csv"
    ts_name = "Timeseries_24.148_120.703_E5_0kWp_crystSi_25_35deg_1deg_2005_2005[UTC+8][daily].csv"
    ts_my = base / "my" / ts_name
    ts_outdir = base / "output" / ts_name

    # fallback: if not found in my/, check output/
    if ts_my.exists():
        timeseries = ts_my
    else:
        alt = base / "output" / ts_name
        if alt.exists():
            timeseries = alt
            print(f"Timeseries found in output/: {timeseries}")
        else:
            print(f"Error: timeseries not found. Checked: {ts_my} and {alt}")
            sys.exit(1)

    if not solar.exists():
        print(f"Error: solar reference not found: {solar}")
        sys.exit(1)

    out_dir = base / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / (timeseries.stem + " [min-max]" + timeseries.suffix)
    params_path = out_dir / "pvgis_stats.json"

    scaler = base / "PVGIS_scaler-min_max.py"
    if not scaler.exists():
        print(f"Error: scaler script not found: {scaler}")
        sys.exit(1)

    cmd = [
        sys.executable,
        str(scaler),
        "-i", str(timeseries),
        "-s", str(solar),
        "-o", str(out_path),
        "-p", str(params_path),
        "-c", "P_Wh",
        "--clip",
    ]

    print("Running scaler:")
    print(" ", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("Scaler failed with return code", e.returncode)
        sys.exit(e.returncode)

    print("Done.")
    print("Output written:", out_path)
    print("Stats JSON:", params_path)


if __name__ == '__main__':
    main()
