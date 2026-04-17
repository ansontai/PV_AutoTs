from __future__ import annotations

import argparse
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Defaults for CLI --input / --output (集中於檔案頂端方便修改)
DEFAULT_INPUT = SCRIPT_DIR / "SolarRecord_260310_1829-daily.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "SolarRecord_260310_1829-daily-1d.csv"

COLUMNS_TO_REMOVE = [
    "INA219_busVoltage_V",
    "INA219_shuntVoltage_mV",
    "INA219_current_mA",
    "INA219_power_mW",
    "ACS712_20A_current_A",
]


def remove_columns(input_path: Path, output_path: Path) -> None:
    if not input_path.is_absolute():
        cwd_candidate = Path.cwd() / input_path
        input_path = cwd_candidate if cwd_candidate.exists() else SCRIPT_DIR / input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    if not output_path.is_absolute():
        output_path = SCRIPT_DIR / output_path

    remove_set = set(COLUMNS_TO_REMOVE)

    with input_path.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header row.")

        kept_fields = [name for name in reader.fieldnames if name not in remove_set]

        with output_path.open("w", encoding="utf-8-sig", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=kept_fields, extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                writer.writerow(row)

    removed_existing = [name for name in COLUMNS_TO_REMOVE if name in (reader.fieldnames or [])]
    missing_columns = [name for name in COLUMNS_TO_REMOVE if name not in (reader.fieldnames or [])]

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Removed columns found in file: {len(removed_existing)}")
    print(f"Columns not found (skipped): {len(missing_columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove weather columns from a CSV file for Wh modeling."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV path (default: {DEFAULT_INPUT.name})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT.name})",
    )
    args = parser.parse_args()

    remove_columns(args.input, args.output)
