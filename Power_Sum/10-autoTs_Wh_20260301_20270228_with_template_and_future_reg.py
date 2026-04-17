from pathlib import Path

import pandas as pd
from autots import AutoTS


BASE_DIR = Path(__file__).resolve().parent
TRAIN_CSV_PATH = BASE_DIR / "SolarRecord(260204)_d_forWh_WithCodis.csv"
FUTURE_REG_CSV_PATH = BASE_DIR / "forecast_weather_1y.csv"
TEMPLATE_PATH = BASE_DIR / "autoTs_template_260310_0139" / "autoTs_template_90d.json"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "forecast_Wh_20260301_20270228_autots_template90d_futureReg.csv"
ROLLING_FORECAST_LENGTH = 90

# User-specified stable model list
DEFAULT_MODEL_LIST = [
    "Theta",
    "ARIMA",
    "RollingRegression",
    "WindowRegression",
    "DatepartRegression",
]

FORECAST_START = pd.Timestamp("2026-03-01")
FORECAST_END = pd.Timestamp("2027-02-28")
def _read_train_target_and_reg(train_csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv_path}")

    df = pd.read_csv(train_csv_path)
    date_col = "Date" if "Date" in df.columns else ("LocalTime" if "LocalTime" in df.columns else None)
    if date_col is None or "Wh" not in df.columns:
        raise ValueError("Training CSV must contain date column ('Date' or 'LocalTime') and 'Wh' column")

    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    target = df[["Date", "Wh"]].copy()
    target["Wh"] = pd.to_numeric(target["Wh"], errors="coerce")

    regressor_exclude = {"Date", "LocalTime", "Wh"}
    reg_cols = [c for c in df.columns if c not in regressor_exclude]
    reg = df[["Date"] + reg_cols].copy()
    for col in reg_cols:
        reg[col] = pd.to_numeric(reg[col], errors="coerce")

    return target, reg


def _read_future_regressor(future_reg_csv_path: Path) -> pd.DataFrame:
    if not future_reg_csv_path.exists():
        raise FileNotFoundError(f"Future regressor CSV not found: {future_reg_csv_path}")

    reg = pd.read_csv(future_reg_csv_path)
    if "Date" not in reg.columns:
        raise ValueError("Future regressor CSV must contain 'Date' column")

    reg["Date"] = pd.to_datetime(reg["Date"], errors="coerce")
    reg = reg.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    reg_cols = [c for c in reg.columns if c != "Date"]
    for col in reg_cols:
        reg[col] = pd.to_numeric(reg[col], errors="coerce")

    return reg


def main() -> None:
    target_df, train_reg_df = _read_train_target_and_reg(TRAIN_CSV_PATH)
    future_reg_df = _read_future_regressor(FUTURE_REG_CSV_PATH)

    forecast_dates = pd.date_range(FORECAST_START, FORECAST_END, freq="D")
    horizon = len(forecast_dates)

    train_mask = (target_df["Date"] < FORECAST_START) & target_df["Wh"].notna()
    train_target = target_df.loc[train_mask, ["Date", "Wh"]].copy()
    if train_target.empty:
        raise ValueError("No usable training target before forecast start date")

    train_target = train_target.set_index("Date").asfreq("D")
    train_target["Wh"] = train_target["Wh"].ffill().bfill()

    train_reg_df = train_reg_df.set_index("Date").sort_index().asfreq("D")
    future_reg_df = future_reg_df.set_index("Date").sort_index().asfreq("D")

    common_reg_cols = sorted(set(train_reg_df.columns).intersection(future_reg_df.columns))
    if not common_reg_cols:
        raise ValueError("No shared regressor columns between training CSV and future regressor CSV")

    train_reg = train_reg_df[common_reg_cols].reindex(train_target.index).ffill().bfill()
    future_reg = future_reg_df[common_reg_cols].reindex(forecast_dates).ffill().bfill()

    if future_reg.isna().all(axis=None):
        raise ValueError("Future regressor has no valid values in forecast window")

    if train_reg.isna().all(axis=None):
        raise ValueError("Training regressor has no valid values after alignment")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE_PATH}")

    history_target = train_target[["Wh"]].copy()
    history_reg = train_reg.copy()
    forecast_chunks: list[pd.DataFrame] = []

    for i in range(0, horizon, ROLLING_FORECAST_LENGTH):
        chunk_dates = forecast_dates[i : i + ROLLING_FORECAST_LENGTH]
        chunk_len = len(chunk_dates)
        chunk_future_reg = future_reg.reindex(chunk_dates).ffill().bfill()

        print(f"\n--- Chunk {i//ROLLING_FORECAST_LENGTH + 1} start -> {chunk_dates[0].date()} to {chunk_dates[-1].date()} (len={chunk_len})")
        print(f"History target range: {history_target.index.min().date()} -> {history_target.index.max().date()} (len={len(history_target)})")

        model = AutoTS(
            forecast_length=chunk_len,
            frequency="D",
            model_list=DEFAULT_MODEL_LIST,
            transformer_list=[
                  "DifferencedTransformer", # 避免被「抹平」成水平線
                  "Scaler", # 避免被「抹平」成水平線
                  ],
            # n_jobs=4,
            n_jobs=-1,
            max_generations=1,
            num_validations=0,
            # min_allowed_train_percent=0.33,
            no_negatives=True,
        )
        model.import_template(str(TEMPLATE_PATH))

        model = model.fit(history_target, future_regressor=history_reg)
        print(f"Model fitted for chunk {i//ROLLING_FORECAST_LENGTH + 1}")
        try:
            prediction = model.predict(forecast_length=chunk_len, future_regressor=chunk_future_reg)
            chunk_forecast = prediction.forecast.copy()
        except Exception as e:
            print(f"ERROR during predict for chunk {i//ROLLING_FORECAST_LENGTH + 1}: {e}")
            # fallback: use recent mean or last value to fill chunk
            if len(history_target) >= 30:
                fill_val = float(history_target["Wh"].iloc[-30:].mean())
            else:
                fill_val = float(history_target["Wh"].iloc[-1])
            chunk_forecast = pd.DataFrame({"Wh": [fill_val] * chunk_len}, index=chunk_dates)
            print(f"Filled chunk with fallback value {fill_val}")

        if "Wh" not in chunk_forecast.columns and len(chunk_forecast.columns) > 0:
            chunk_forecast = chunk_forecast.rename(columns={chunk_forecast.columns[0]: "Wh"})
        if "Wh" not in chunk_forecast.columns:
            raise ValueError("Prediction output does not include 'Wh' column")

        chunk_forecast = chunk_forecast.reindex(chunk_dates)[["Wh"]]
        if chunk_forecast["Wh"].isna().any():
            missing = int(chunk_forecast["Wh"].isna().sum())
            print(f"WARNING: Chunk {i//ROLLING_FORECAST_LENGTH + 1} has {missing} missing forecast values; filling with recent mean/last value")
            if len(history_target) >= 30:
                fill_val = float(history_target["Wh"].iloc[-30:].mean())
            else:
                fill_val = float(history_target["Wh"].iloc[-1])
            chunk_forecast["Wh"] = chunk_forecast["Wh"].fillna(fill_val)
            print(f"Filled {missing} missing entries with {fill_val}")

        forecast_chunks.append(chunk_forecast)

        history_target = pd.concat([history_target, chunk_forecast]).sort_index()
        history_target = history_target[~history_target.index.duplicated(keep="last")]

        history_reg = pd.concat([history_reg, chunk_future_reg]).sort_index()
        history_reg = history_reg[~history_reg.index.duplicated(keep="last")]

    forecast = pd.concat(forecast_chunks).sort_index().reindex(forecast_dates)

    out = (
        forecast.reindex(forecast_dates)[["Wh"]]
        .reset_index()
        .rename(columns={"index": "Date", "Wh": "Wh_pred"})
    )

    if out["Wh_pred"].isna().any():
        missing = int(out["Wh_pred"].isna().sum())
        print(f"FINAL WARNING: Forecast contains {missing} missing values in requested date range")
        print(out[out["Wh_pred"].isna()])
        # write a diagnostic file with missing dates
        diag_path = OUTPUT_DIR / "forecast_missing_dates_debug.csv"
        out[out["Wh_pred"].isna()].to_csv(diag_path, index=False)
        print(f"Wrote missing-date diagnostic: {diag_path}")

    out.to_csv(OUTPUT_FILE, index=False)

    print(f"Train CSV: {TRAIN_CSV_PATH}")
    print(f"Future regressor CSV: {FUTURE_REG_CSV_PATH}")
    print(f"Template: {TEMPLATE_PATH}")
    print(f"Training range: {train_target.index.min().date()} -> {train_target.index.max().date()}")
    print(f"Forecast range: {FORECAST_START.date()} -> {FORECAST_END.date()} ({horizon} days)")
    print(f"Regressor columns used: {len(common_reg_cols)}")
    print(f"Rolling forecast chunk size: {ROLLING_FORECAST_LENGTH}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
