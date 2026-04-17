import gc
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from autots import AutoTS


# ====== Base paths ======
THIS_FILE = Path(__file__).resolve()
BASE_DIR = THIS_FILE.parent                      # .../Power_Sum
PROJECT_ROOT = BASE_DIR.parent                   
ROOT_OUTPUT_DIR = THIS_FILE.parent / "output"

# ====== Input files ======
TRAIN_CSV_PATH = BASE_DIR / "SolarRecord(260204)_d_forWh_WithCodis.csv"
FUTURE_REG_CSV_PATH = BASE_DIR / "forecast_weather_1y.csv"
TEMPLATE_PATH = BASE_DIR / "autoTs_template_260310_0139" / "autoTs_template_90d.json"

# ====== Forecast config ======
BASE_NAME = "forecast_Wh_20260301_20270228_autots_template90d_futureReg"
FORECAST_START = pd.Timestamp("2026-02-05")
FORECAST_END = pd.Timestamp("2027-02-28")
ROLLING_FORECAST_LENGTH = 90
DEFAULT_MAX_GENERATIONS = 6  # 可調整以平衡速度與預測品質

DEFAULT_MODEL_LIST = [
    "Theta",
    "ARIMA",
    "RollingRegression",
    "WindowRegression",
    "DatepartRegression",
    # add 
    #X 'SeasonalNaive', 
]

MainLoop = True  # 設為 False 可停止無限執行

def make_run_dir(script_file: Path, loop_idx: int = 0) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{script_file.stem}_{ts}_{loop_idx}"
    run_dir = ROOT_OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _read_train_target_and_reg(train_csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv_path}")

    df = pd.read_csv(train_csv_path)
    date_col = "Date" if "Date" in df.columns else ("LocalTime" if "LocalTime" in df.columns else None)
    if date_col is None or "Wh" not in df.columns:
        raise ValueError("Training CSV must contain Date/LocalTime and Wh columns")

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
        raise ValueError("Future regressor CSV must contain Date column")

    reg["Date"] = pd.to_datetime(reg["Date"], errors="coerce")
    reg = reg.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    reg_cols = [c for c in reg.columns if c != "Date"]
    for col in reg_cols:
        reg[col] = pd.to_numeric(reg[col], errors="coerce")

    return reg


def run_forecast(run_dir: Path) -> Path:
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
        raise ValueError("No shared regressor columns between training and future regressor CSV")

    train_reg = train_reg_df[common_reg_cols].reindex(train_target.index).ffill().bfill()
    future_reg = future_reg_df[common_reg_cols].reindex(forecast_dates).ffill().bfill()

    if future_reg.isna().all(axis=None):
        raise ValueError("Future regressor has no valid values in forecast window")
    if train_reg.isna().all(axis=None):
        raise ValueError("Training regressor has no valid values after alignment")
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE_PATH}")

    history_target = train_target[["Wh"]].copy()
    history_reg = train_reg.copy()
    forecast_chunks: list[pd.DataFrame] = []

    for i in range(0, horizon, ROLLING_FORECAST_LENGTH):
        chunk_dates = forecast_dates[i:i + ROLLING_FORECAST_LENGTH]
        chunk_len = len(chunk_dates)
        chunk_future_reg = future_reg.reindex(chunk_dates).ffill().bfill()

        print(f"\n--- Chunk {i // ROLLING_FORECAST_LENGTH + 1}: {chunk_dates[0].date()} -> {chunk_dates[-1].date()} (len={chunk_len})")

        model = AutoTS(
            forecast_length=chunk_len,
            frequency="D",
            model_list=DEFAULT_MODEL_LIST,
            transformer_list=["DifferencedTransformer", "Scaler"],
            n_jobs=-1,
            max_generations=DEFAULT_MAX_GENERATIONS,
            # num_validations=0,
            num_validations=1,
            no_negatives=True,
        )
        model.import_template(str(TEMPLATE_PATH))
        model = model.fit(history_target, future_regressor=history_reg)

        try:
            prediction = model.predict(forecast_length=chunk_len, future_regressor=chunk_future_reg)
            chunk_forecast = prediction.forecast.copy()
        except Exception as e:
            print(f"Predict failed in chunk {i // ROLLING_FORECAST_LENGTH + 1}: {e}")
            fill_val = float(history_target["Wh"].iloc[-30:].mean()) if len(history_target) >= 30 else float(history_target["Wh"].iloc[-1])
            chunk_forecast = pd.DataFrame({"Wh": [fill_val] * chunk_len}, index=chunk_dates)

        if "Wh" not in chunk_forecast.columns and len(chunk_forecast.columns) > 0:
            chunk_forecast = chunk_forecast.rename(columns={chunk_forecast.columns[0]: "Wh"})
        if "Wh" not in chunk_forecast.columns:
            raise ValueError("Prediction output does not include Wh column")

        chunk_forecast = chunk_forecast.reindex(chunk_dates)[["Wh"]]
        if chunk_forecast["Wh"].isna().any():
            fill_val = float(history_target["Wh"].iloc[-30:].mean()) if len(history_target) >= 30 else float(history_target["Wh"].iloc[-1])
            chunk_forecast["Wh"] = chunk_forecast["Wh"].fillna(fill_val)

        forecast_chunks.append(chunk_forecast)

        history_target = pd.concat([history_target, chunk_forecast]).sort_index()
        history_target = history_target[~history_target.index.duplicated(keep="last")]

        history_reg = pd.concat([history_reg, chunk_future_reg]).sort_index()
        history_reg = history_reg[~history_reg.index.duplicated(keep="last")]

    forecast = pd.concat(forecast_chunks).sort_index().reindex(forecast_dates)
    out = forecast.reset_index().rename(columns={"index": "Date", "Wh": "Wh_pred"})

    raw_csv = run_dir / f"{BASE_NAME}.csv"
    out.to_csv(raw_csv, index=False)
    print(f"Forecast CSV: {raw_csv}")
    return raw_csv


def fill_missing_forecast(raw_csv: Path, run_dir: Path) -> Path:
    if not raw_csv.exists():
        raise FileNotFoundError(f"Output forecast not found: {raw_csv}")
    if not TRAIN_CSV_PATH.exists():
        raise FileNotFoundError(f"Training CSV not found: {TRAIN_CSV_PATH}")

    out = pd.read_csv(raw_csv)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    train = pd.read_csv(TRAIN_CSV_PATH)
    date_col = "Date" if "Date" in train.columns else ("LocalTime" if "LocalTime" in train.columns else None)
    if date_col is None:
        raise ValueError("Training CSV missing Date/LocalTime column")
    if "Wh" not in train.columns:
        raise ValueError("Training CSV missing Wh column")

    train["Date"] = pd.to_datetime(train[date_col], errors="coerce")
    train = train.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    train["Wh"] = pd.to_numeric(train["Wh"], errors="coerce")
    train = train.set_index("Date")[["Wh"]]

    combined_index = pd.date_range(train.index.min(), FORECAST_END, freq="D")
    combined = pd.Series(index=combined_index, dtype="float64")
    combined.update(train["Wh"].reindex(combined.index))
    combined.update(out["Wh_pred"].rename("Wh").reindex(combined.index))

    forecast_dates = pd.date_range(FORECAST_START, FORECAST_END, freq="D")
    missing_before = int(out["Wh_pred"].isna().sum())
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
    filled_csv = run_dir / f"{BASE_NAME}_filled.csv"
    out.to_csv(filled_csv, index=True, index_label="Date")

    print(f"Original missing: {missing_before}, Filled: {filled_count}")
    print(f"Filled CSV: {filled_csv}")
    return filled_csv


def compute_and_save_metrics(y_true, y_pred, train_df: pd.DataFrame, run_dir: Path, forecast_start: pd.Timestamp, forecast_end: pd.Timestamp) -> Path | None:
    """Compute forecast metrics and save JSON to `run_dir`.

    Returns the Path to the written JSON or None on failure.
    """
    try:
        import numpy as _np
        from sklearn.metrics import mean_absolute_error as _mae, r2_score as _r2

        y_true = _np.asarray(y_true, dtype=float)
        y_pred = _np.asarray(y_pred, dtype=float)

        # training series: values before forecast start
        if train_df is None or train_df.empty:
            train_series = _np.asarray([], dtype=float)
        else:
            try:
                train_series = train_df[train_df["Date"] < forecast_start]["Wh"].dropna().values
            except Exception:
                # fallback: any Wh values in train_df
                train_series = train_df["Wh"].dropna().values if "Wh" in train_df.columns else _np.asarray([], dtype=float)

        mae = float(_mae(y_true, y_pred))
        denom = _np.mean(_np.abs(_np.diff(train_series))) if train_series.size > 1 else 0.0
        mase = float(mae / denom) if denom != 0 else _np.nan

        rmse = float(_np.sqrt(_np.mean((y_pred - y_true) ** 2)))
        denom_rmsse = float(_np.sqrt(_np.mean(_np.diff(train_series) ** 2))) if train_series.size > 1 else 0.0
        rmsse = float(rmse / denom_rmsse) if denom_rmsse != 0 else _np.nan

        mean_actual = float(_np.mean(y_true)) if y_true.size > 0 else 0.0
        nmae = float(mae / mean_actual) if mean_actual != 0 else _np.nan
        nrmse = float(rmse / mean_actual) if mean_actual != 0 else _np.nan

        smape = float(_np.mean(2.0 * _np.abs(y_pred - y_true) / (_np.abs(y_true) + _np.abs(y_pred) + 1e-9)) * 100)

        nonzero_mask = _np.abs(y_true) > 1e-9
        if nonzero_mask.any():
            mape = float(_np.mean(_np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100)
        else:
            mape = _np.nan

        r2 = float(_r2(y_true, y_pred))

        scores = {
            'MAE': mae,
            'MASE_lag1': float(mase) if not _np.isnan(mase) else None,
            'RMSSE': float(rmsse) if not _np.isnan(rmsse) else None,
            'nMAE': float(nmae) if not _np.isnan(nmae) else None,
            'nRMSE': float(nrmse) if not _np.isnan(nrmse) else None,
            'MAPE(%)': float(mape) if not _np.isnan(mape) else None,
            'SMAPE(%)': smape,
            'R2': r2,
        }

        horizon = len(pd.date_range(forecast_start, forecast_end, freq='D'))
        out_json = run_dir / f'forecast_Wh_metrics_{horizon}d.json'
        # convert numpy types to python floats/None for JSON
        scores_clean = {k: (None if v is None else float(v)) for k, v in scores.items()}
        with open(out_json, 'w', encoding='utf-8') as jf:
            json.dump(scores_clean, jf, ensure_ascii=False, indent=2)

        return out_json
    except Exception as e:
        print('compute_and_save_metrics failed:', e)
        return None


def plot_forecast_comparison(filled_or_raw_csv: Path, run_dir: Path) -> Path:
    fc = pd.read_csv(filled_or_raw_csv)
    if "Date" not in fc.columns:
        raise ValueError("Forecast CSV must include Date column")
    fc["Date"] = pd.to_datetime(fc["Date"], errors="coerce")
    fc = fc.dropna(subset=["Date"]).sort_values("Date")

    if "Wh_pred" not in fc.columns:
        if "Wh" in fc.columns:
            fc = fc.rename(columns={"Wh": "Wh_pred"})
        else:
            num_cols = [c for c in fc.columns if c != "Date"]
            if not num_cols:
                raise ValueError("Forecast CSV has no forecast numeric column")
            fc = fc.rename(columns={num_cols[0]: "Wh_pred"})

    if not TRAIN_CSV_PATH.exists():
        train = pd.DataFrame(columns=["Date", "Wh"])
    else:
        train = pd.read_csv(TRAIN_CSV_PATH)
        date_col = "Date" if "Date" in train.columns else ("LocalTime" if "LocalTime" in train.columns else None)
        if date_col is None or "Wh" not in train.columns:
            train = pd.DataFrame(columns=["Date", "Wh"])
        else:
            train["Date"] = pd.to_datetime(train[date_col], errors="coerce")
            train = train.dropna(subset=["Date"])[["Date", "Wh"]]
            train["Wh"] = pd.to_numeric(train["Wh"], errors="coerce")

    forecast_dates = pd.date_range(FORECAST_START, FORECAST_END, freq="D")
    fc = fc.set_index("Date").reindex(forecast_dates).reset_index().rename(columns={"index": "Date"})

    merged = fc.merge(train, on="Date", how="left")
    y_pred = merged["Wh_pred"].astype(float).values
    y_true = merged["Wh"].astype(float).values if "Wh" in merged.columns else np.array([np.nan] * len(merged))

    last_hist = None
    if not train.empty:
        last_hist_vals = train[train["Date"] < FORECAST_START]["Wh"].dropna()
        if not last_hist_vals.empty:
            last_hist = float(last_hist_vals.iloc[-1])

    naive = []
    prev = last_hist if last_hist is not None else np.nan
    for i in range(len(merged)):
        naive.append(prev)
        if not np.isnan(y_true[i]):
            prev = float(y_true[i])

    # Keep same style as your original (only AutoTS line)
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_dates, y_pred, label="AutoTS Forecast", linewidth=2)
    plt.title("Wh: AutoTS vs Actual vs Naive (2026-03-01 -> 2027-02-28)")
    plt.xlabel("Date")
    plt.ylabel("Wh")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_png = run_dir / f"{BASE_NAME}_comparison.png"
    plt.savefig(plot_png, dpi=150)
    plt.close()

    print(f"Plot PNG: {plot_png}")
    # compute metrics and save JSON using helper
    try:
        out_json = compute_and_save_metrics(y_true, y_pred, train, run_dir, FORECAST_START, FORECAST_END)
        if out_json is not None:
            print('Metrics JSON saved to', out_json)
    except Exception as e:
        print('Failed to compute/save metrics:', e)

    return plot_png


def find_wh_column(df: pd.DataFrame) -> str | None:
    for name in ("Wh_pred", "Wh"):
        if name in df.columns:
            return name
    for col in df.columns:
        if col.lower() == "date":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def sum_forecast_wh(filled_or_raw_csv: Path, run_dir: Path) -> Path:
    df = pd.read_csv(filled_or_raw_csv)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    wh_col = find_wh_column(df)
    if wh_col is None:
        raise ValueError("Could not identify a numeric Wh column in forecast CSV")

    df[wh_col] = pd.to_numeric(df[wh_col], errors="coerce")
    valid = df[wh_col].dropna()

    total_wh = float(valid.sum()) if not valid.empty else 0.0
    count_days = int(valid.shape[0])

    out = pd.DataFrame([{
        "forecast_file": filled_or_raw_csv.name,
        "wh_column": wh_col,
        "forecast_start": str(df["Date"].min()) if "Date" in df.columns else "",
        "forecast_end": str(df["Date"].max()) if "Date" in df.columns else "",
        "days_counted": count_days,
        "total_wh": total_wh,
        "generated_at": datetime.now().isoformat(),
    }])

    sum_csv = run_dir / f"{BASE_NAME}_sum.csv"
    out.to_csv(sum_csv, index=False)
    print(f"Sum CSV: {sum_csv} (total_wh={total_wh}, days={count_days})")
    return sum_csv


def main(loop_idx: int = 0) -> None:
    run_dir = make_run_dir(THIS_FILE, loop_idx)
    print(f"Run output folder: {run_dir}")

    raw_csv = run_forecast(run_dir)
    filled_csv = fill_missing_forecast(raw_csv, run_dir)
    plot_png = plot_forecast_comparison(filled_csv, run_dir)
    sum_csv = sum_forecast_wh(filled_csv, run_dir)

    # 顯式釋放大物件並強制垃圾回收
    del raw_csv, filled_csv, plot_png, sum_csv
    gc.collect()

    print("\nAll done.")
    # ...existing code...


if __name__ == "__main__":
    loop_idx = 0
    while MainLoop:
        main(loop_idx)
        loop_idx += 1
        # 這裡可加上 break 條件或 sleep 等待