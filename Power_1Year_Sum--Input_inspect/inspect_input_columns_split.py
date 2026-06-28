from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = INPUT_DIR / "SolarRecord(260228)_d_forWh_WithCodis[date].csv"
TMY_CSV = INPUT_DIR / "tmy_24.148_120.703_2005_2023[UTC+8][daily][mapped][dateAdj].csv"

DATE_CANDIDATES = ("date", "Date", "LocalTime")
TARGET_COLUMN = "Wh"
FIT_REGRESSOR_COLUMNS = ["Temperature", "RH"]


def detect_date_column(cols):
    for c in DATE_CANDIDATES:
        if c in cols:
            return c
    return None


def inspect_and_save(path: Path, out_name: str):
    rows = []
    if not path.exists():
        rows.append({"file": path.name, "column": None, "used": False, "usage_type": "file_missing"})
        df_out = pd.DataFrame(rows)
        df_out.to_csv(OUTPUT_DIR / out_name, index=False, encoding="utf-8-sig")
        return

    df0 = pd.read_csv(path, nrows=0)
    cols = list(df0.columns)
    date_col = detect_date_column(cols)

    for c in cols:
        if c == TARGET_COLUMN:
            usage = "target"
            used = True
        elif c in FIT_REGRESSOR_COLUMNS:
            usage = "regressor"
            used = True
        elif c == date_col:
            usage = "date"
            used = True
        else:
            usage = "unused"
            used = False
        rows.append({"file": path.name, "column": c, "used": used, "usage_type": usage})

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / out_name, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    inspect_and_save(TRAIN_CSV, "SolarRecord_columns.csv")
    inspect_and_save(TMY_CSV, "TMY_columns.csv")
    print("Wrote:", OUTPUT_DIR / "SolarRecord_columns.csv")
    print("Wrote:", OUTPUT_DIR / "TMY_columns.csv")
