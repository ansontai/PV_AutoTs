from pathlib import Path
import pandas as pd
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prefer the filled forecast file produced by 10b; fall back to unfilled
BASE_NAME = "forecast_Wh_20260301_20270228_autots_template90d_futureReg"
FILLED_CSV = OUTPUT_DIR / f"{BASE_NAME}_filled.csv"
RAW_CSV = OUTPUT_DIR / f"{BASE_NAME}.csv"
OUT_SUM_CSV = OUTPUT_DIR / f"{BASE_NAME}_sum.csv"


def find_wh_column(df: pd.DataFrame) -> str | None:
    # common names
    for name in ("Wh_pred", "Wh"):
        if name in df.columns:
            return name
    # fallback: first numeric column other than Date
    for col in df.columns:
        if col.lower() == "date":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def main():
    src = FILLED_CSV if FILLED_CSV.exists() else RAW_CSV
    if not src.exists():
        raise FileNotFoundError(f"No forecast CSV found. Looking for {FILLED_CSV} or {RAW_CSV}")

    df = pd.read_csv(src)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    wh_col = find_wh_column(df)
    if wh_col is None:
        raise ValueError("Could not identify a numeric Wh column in the forecast CSV")

    df[wh_col] = pd.to_numeric(df[wh_col], errors="coerce")
    valid = df[wh_col].dropna()
    total_wh = float(valid.sum()) if not valid.empty else 0.0
    count_days = int(valid.shape[0])

    out = pd.DataFrame([
        {
            "forecast_file": src.name,
            "wh_column": wh_col,
            "forecast_start": str(df["Date"].min()) if "Date" in df.columns else "",
            "forecast_end": str(df["Date"].max()) if "Date" in df.columns else "",
            "days_counted": count_days,
            "total_wh": total_wh,
            "generated_at": datetime.now().isoformat(),
        }
    ])

    out.to_csv(OUT_SUM_CSV, index=False)
    print(f"Wrote sum to: {OUT_SUM_CSV} (total_wh={total_wh}, days={count_days})")


if __name__ == "__main__":
    main()
