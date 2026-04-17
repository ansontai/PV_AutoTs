import os
from pathlib import Path

import pandas as pd
from autots import AutoTS


CSV_PATH = Path(r"t:\OneDrive\1TB\School\python_local\Power_Sum\SolarRecord(260204)_d_forWh_WithCodis-Add1yWeather.csv")
TEMPLATE_PATH = Path(r"t:\OneDrive\1TB\School\python_local\Power_Sum\autoTs_template_260310_0139\autoTs_template_90d.json")
OUTPUT_DIR = Path(r"t:\OneDrive\1TB\School\python_local\Power_Sum\output")
OUTPUT_FILE = OUTPUT_DIR / "forecast_Wh_20260301_20270228_autots_template90d.csv"

FORECAST_START = pd.Timestamp("2026-03-01")
FORECAST_END = pd.Timestamp("2027-02-28")
FIT_FORECAST_LENGTH = 90


def load_and_prepare(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Date" not in df.columns or "Wh" not in df.columns:
        raise ValueError("CSV must contain 'Date' and 'Wh' columns")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    # Convert all feature columns to numeric and keep only weather regressors.
    feature_cols = [c for c in df.columns if c not in {"Date", "LocalTime", "Wh"}]
    features = df[["Date"] + feature_cols].copy()
    for col in feature_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    target = df[["Date", "Wh"]].copy()
    target["Wh"] = pd.to_numeric(target["Wh"], errors="coerce")

    return target, features


def main() -> None:
    target_df, feature_df = load_and_prepare(CSV_PATH)

    forecast_dates = pd.date_range(FORECAST_START, FORECAST_END, freq="D")
    horizon = len(forecast_dates)

    # Train on all known Wh values before forecast start.
    train_mask = (target_df["Date"] < FORECAST_START) & target_df["Wh"].notna()
    train_target = target_df.loc[train_mask, ["Date", "Wh"]].copy()
    if train_target.empty:
        raise ValueError("No training data available before forecast start date")

    train_target = train_target.set_index("Date").asfreq("D")
    train_target["Wh"] = train_target["Wh"].ffill().bfill()

    feature_df = feature_df.set_index("Date").sort_index().asfreq("D")
    feature_df = feature_df.ffill().bfill()

    train_reg = feature_df.reindex(train_target.index)
    future_reg = feature_df.reindex(forecast_dates)

    if future_reg.isna().all(axis=None):
        raise ValueError("Future regressor data is empty for forecast window")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = AutoTS(
        forecast_length=FIT_FORECAST_LENGTH,
        frequency="D",
        model_list="default",
        n_jobs=4,
        num_validations=0,
        no_negatives=True,
    )

    if TEMPLATE_PATH.exists():
        model.import_template(str(TEMPLATE_PATH))
    else:
        raise FileNotFoundError(f"Template not found: {TEMPLATE_PATH}")

    model = model.fit(train_target[["Wh"]], future_regressor=train_reg)
    prediction = model.predict(forecast_length=horizon, future_regressor=future_reg)
    forecast = prediction.forecast.copy()

    if "Wh" not in forecast.columns:
        forecast = forecast.rename(columns={forecast.columns[0]: "Wh"})

    forecast = forecast.reindex(forecast_dates)
    out = forecast[["Wh"]].reset_index().rename(columns={"index": "Date", "Wh": "Wh_pred"})

    out.to_csv(OUTPUT_FILE, index=False)

    print(f"Training range: {train_target.index.min().date()} -> {train_target.index.max().date()}")
    print(f"Forecast range: {FORECAST_START.date()} -> {FORECAST_END.date()} ({horizon} days)")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
