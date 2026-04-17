from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "output" / "forecast_Wh_20260301_20270228_autots_template90d_futureReg.csv"
TRAIN_CSV_PATH = BASE_DIR / "SolarRecord(260204)_d_forWh_WithCodis.csv"
OUTPUT_FILLED = BASE_DIR / "output" / "forecast_Wh_20260301_20270228_autots_template90d_futureReg_filled.csv"

FORECAST_START = pd.Timestamp("2026-03-01")
FORECAST_END = pd.Timestamp("2027-02-28")


def main():
    if not OUTPUT_FILE.exists():
        raise FileNotFoundError(f"Output forecast not found: {OUTPUT_FILE}")
    if not TRAIN_CSV_PATH.exists():
        raise FileNotFoundError(f"Training CSV not found: {TRAIN_CSV_PATH}")

    out = pd.read_csv(OUTPUT_FILE)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    # load training history to provide prior values for filling
    train = pd.read_csv(TRAIN_CSV_PATH)
    date_col = "Date" if "Date" in train.columns else ("LocalTime" if "LocalTime" in train.columns else None)
    if date_col is None:
        raise ValueError("Training CSV missing Date/LocalTime column")
    train["Date"] = pd.to_datetime(train[date_col], errors="coerce")
    train = train.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    if "Wh" not in train.columns:
        raise ValueError("Training CSV missing Wh column")
    train["Wh"] = pd.to_numeric(train["Wh"], errors="coerce")
    train = train.set_index("Date")[["Wh"]]

    # build combined series: historical Wh up to forecast start, then predicted values
    combined_index = pd.date_range(train.index.min(), FORECAST_END, freq="D")
    combined = pd.Series(index=combined_index, dtype="float64")
    # fill historical
    combined.update(train["Wh"].reindex(combined.index))
    # fill predictions
    combined.update(out["Wh_pred"].rename("Wh").reindex(combined.index))

    # iterate over forecast dates and fill NaNs
    forecast_dates = pd.date_range(FORECAST_START, FORECAST_END, freq="D")
    missing_before = out["Wh_pred"].isna().sum()
    filled_count = 0
    for d in forecast_dates:
        if pd.isna(combined.at[d]):
            prev = combined.loc[:(d - pd.Timedelta(days=1))].dropna()
            if len(prev) >= 30:
                val = prev.tail(30).mean()
            elif len(prev) >= 1:
                val = prev.iloc[-1]
            else:
                val = 0.0
            combined.at[d] = float(val)
            if d in out.index:
                out.at[d, "Wh_pred"] = float(val)
            filled_count += 1

    out = out.reindex(forecast_dates)
    out.to_csv(OUTPUT_FILLED, index=True, index_label="Date")

    print(f"Original missing: {missing_before}")
    print(f"Filled missing: {filled_count}")
    print(f"Filled output saved to: {OUTPUT_FILLED}")


if __name__ == '__main__':
    main()
