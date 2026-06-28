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


def inspect_columns(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        rows.append({"file": path.name, "column": None, "used": False, "usage_type": "file_missing"})
        return rows

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
    return rows


def inspect_and_save_split(train_path: Path, tmy_path: Path) -> None:
    train_rows = inspect_columns(train_path)
    tmy_rows = inspect_columns(tmy_path)

    train_cols = {r['column'] for r in train_rows if r['column'] is not None}
    tmy_cols = {r['column'] for r in tmy_rows if r['column'] is not None}

    for r in train_rows:
        col = r.get('column')
        r['exists_in_other_file'] = bool(col in tmy_cols) if col is not None else False

    for r in tmy_rows:
        col = r.get('column')
        r['exists_in_other_file'] = bool(col in train_cols) if col is not None else False

    pd.DataFrame(train_rows).to_csv(OUTPUT_DIR / "SolarRecord_columns.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(tmy_rows).to_csv(OUTPUT_DIR / "TMY_columns.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    inspect_and_save_split(TRAIN_CSV, TMY_CSV)
    print("Wrote:", OUTPUT_DIR / "SolarRecord_columns.csv")
    print("Wrote:", OUTPUT_DIR / "TMY_columns.csv")
