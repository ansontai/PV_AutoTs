from __future__ import annotations

import pickle
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from autots import AutoTS

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:
    matplotlib = None
    mdates = None
    plt = None
    MATPLOTLIB_AVAILABLE = False


FORECAST_LENGTH = 365
FREQUENCY = "D"
PREDICTION_INTERVAL = 0.9
# MAX_GENERATIONS = 15
MAX_GENERATIONS = 30
NUM_VALIDATIONS = 3
VALIDATION_METHOD = "backwards"
ENSEMBLE = "all"
NO_NEGATIVES = True

TARGET_COLUMN = "Wh"
DATE_CANDIDATES = ("date", "Date", "LocalTime")
FIT_REGRESSOR_COLUMNS = ["Temperature", "RH"]

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
TRAIN_CSV = INPUT_DIR / "SolarRecord(260228)_d_forWh_WithCodis[date].csv"
TMY_CSV = INPUT_DIR / "tmy_24.148_120.703_2005_2023[UTC+8][daily][mapped][dateAdj].csv"


@dataclass
class RunArtifacts:
    timestamp: str
    run_output_dir: Path
    forecast_csv: Path
    metrics_csv: Path
    log_txt: Path
    model_pickle: Path


@dataclass
class HoldoutBacktestResult:
    metrics: dict[str, float]
    holdout_length: int
    actual: pd.Series | None = None
    forecast: pd.Series | None = None
    lastvalue: pd.Series | None = None


class Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line)
        self.lines.append(line)

    def dump(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def detect_date_column(columns: Iterable[str]) -> str:
    for col in DATE_CANDIDATES:
        if col in columns:
            return col
    raise ValueError(f"No date column found. Expected one of {DATE_CANDIDATES}.")


def read_train_data(csv_path: Path, logger: Logger) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    date_col = detect_date_column(df.columns)
    required = [TARGET_COLUMN] + FIT_REGRESSOR_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Training CSV missing required columns: {missing}")

    out = df[[date_col, TARGET_COLUMN] + FIT_REGRESSOR_COLUMNS].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    for col in [TARGET_COLUMN] + FIT_REGRESSOR_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=[date_col]).sort_values(date_col)
    out = out.drop_duplicates(subset=[date_col], keep="last")
    out = out.rename(columns={date_col: "date"}).set_index("date")
    out = out.asfreq(FREQUENCY)

    # Keep target strict; regressors are gap-filled for model stability.
    out[TARGET_COLUMN] = out[TARGET_COLUMN].interpolate(limit_direction="both")
    out[FIT_REGRESSOR_COLUMNS] = out[FIT_REGRESSOR_COLUMNS].ffill().bfill()
    out = out.dropna(subset=[TARGET_COLUMN])

    logger.info(
        "Loaded training data: "
        f"rows={len(out)}, date_range={out.index.min().date()} -> {out.index.max().date()}"
    )
    return out


def read_tmy_data(csv_path: Path, logger: Logger) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"TMY CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    date_col = detect_date_column(df.columns)
    missing = [c for c in FIT_REGRESSOR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"TMY CSV missing required columns: {missing}")

    out = df[[date_col] + FIT_REGRESSOR_COLUMNS].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    for col in FIT_REGRESSOR_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=[date_col]).sort_values(date_col)
    out = out.drop_duplicates(subset=[date_col], keep="last")
    out = out.rename(columns={date_col: "date"}).set_index("date")
    out = out.asfreq(FREQUENCY)

    logger.info(
        "Loaded TMY data: "
        f"rows={len(out)}, date_range={out.index.min().date()} -> {out.index.max().date()}"
    )
    return out


def build_predict_regressor(
    tmy_df: pd.DataFrame,
    forecast_index: pd.DatetimeIndex,
    logger: Logger,
) -> pd.DataFrame:
    pred_fr = tmy_df.reindex(forecast_index)[FIT_REGRESSOR_COLUMNS].copy()

    if pred_fr.isna().any().any():
        logger.info("TMY has missing values on forecast dates; filling by month-day climatology.")
        tmy_tmp = tmy_df[FIT_REGRESSOR_COLUMNS].copy()
        tmy_tmp["md"] = tmy_tmp.index.strftime("%m-%d")
        by_md = tmy_tmp.groupby("md")[FIT_REGRESSOR_COLUMNS].mean()

        md_keys = forecast_index.strftime("%m-%d")
        fill_by_md = by_md.reindex(md_keys)
        fill_by_md.index = forecast_index

        for col in FIT_REGRESSOR_COLUMNS:
            pred_fr[col] = pred_fr[col].fillna(fill_by_md[col])

    pred_fr = pred_fr.ffill().bfill()

    if pred_fr.isna().any().any():
        raise ValueError("Predict future_regressor still has NaN after fallback fill.")

    return pred_fr


def save_forecast_vs_actual_vs_lastvalue_plot(
    plot_path: Path,
    actual: pd.Series,
    forecast: pd.Series,
    lastvalue: pd.Series,
    title: str,
    logger: Logger,
    metrics: dict[str, float] | None = None,
) -> bool:
    if not MATPLOTLIB_AVAILABLE:
        logger.info(f"Matplotlib not available; skipping plot: {plot_path}")
        return False

    actual = actual.astype(float)
    forecast = forecast.astype(float)
    lastvalue = lastvalue.astype(float)
    fig, ax = plt.subplots(figsize=(6, 3), dpi=300, constrained_layout=True)
    ax.plot(actual.index, actual.values, label="Actual", color="black", linewidth=2.5)
    ax.plot(forecast.index, forecast.values, label="AutoTS Forecast", color="dimgray", linewidth=2.5)
    ax.plot(lastvalue.index, lastvalue.values, label="LastValueNaive", color="gray", linewidth=2, linestyle="--")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel(TARGET_COLUMN)
    ax.grid(alpha=0.35, linestyle=":", linewidth=0.8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    metrics_lines: list[str] = []
    if metrics:
        if not np.isnan(metrics.get("holdout_mase", np.nan)):
            metrics_lines.append(f"MASE={metrics['holdout_mase']:.3f}")
        if not np.isnan(metrics.get("holdout_rmse", np.nan)):
            metrics_lines.append(f"RMSE={metrics['holdout_rmse']:.3f}")
        if not np.isnan(metrics.get("holdout_smape_pct", np.nan)):
            metrics_lines.append(f"sMAPE={metrics['holdout_smape_pct']:.2f}%")
        if not np.isnan(metrics.get("holdout_mape_pct", np.nan)):
            metrics_lines.append(f"MAPE={metrics['holdout_mape_pct']:.2f}%")

    if metrics_lines:
        ax.text(
            0.98,
            0.98,
            "\n".join(metrics_lines),
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="right",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
        )

    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 0.60), fontsize=9, frameon=False)
    fig.subplots_adjust(bottom=0.18, right=0.80)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return True


def save_future_forecast_plot(
    plot_path: Path,
    forecast_index: pd.DatetimeIndex,
    forecast_values: np.ndarray,
    lower_values: np.ndarray | None = None,
    upper_values: np.ndarray | None = None,
    title: str = "365-Day Forecast",
    logger: Logger | None = None,
) -> bool:
    """Save a future forecast plot with optional confidence bounds (no actual data)."""
    if not MATPLOTLIB_AVAILABLE:
        if logger:
            logger.info(f"Matplotlib not available; skipping plot: {plot_path}")
        return False
    fig, ax = plt.subplots(figsize=(6, 3), dpi=300, constrained_layout=True)
    ax.plot(forecast_index, forecast_values, label="AutoTS Forecast", color="dimgray", linewidth=2.5)

    if lower_values is not None and upper_values is not None:
        ax.fill_between(
            forecast_index,
            lower_values,
            upper_values,
            alpha=0.2,
            color="dimgray",
            label="90% Prediction Interval",
        )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel(TARGET_COLUMN)
    ax.grid(alpha=0.35, linestyle=":", linewidth=0.8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return True


def extract_best_validation_metrics(model: AutoTS) -> dict[str, float | str]:
    result = {
        "autots_score": np.nan,
        "validation_smape": np.nan,
        "validation_mae": np.nan,
        "validation_rmse": np.nan,
        "best_model": "",
    }

    for mode in ("validation", None):
        try:
            df = model.results(mode) if mode else model.results()
            if not hasattr(df, "columns") or len(df) == 0:
                continue

            score_col_candidates = [
                "Validation Score",
                "validation_score",
                "Score",
                "score",
                "smape_weighted",
            ]
            score_col = next((c for c in score_col_candidates if c in df.columns), None)
            if score_col is None:
                score_col = df.columns[0]

            numeric_score = pd.to_numeric(df[score_col], errors="coerce")
            if numeric_score.notna().any():
                best_idx = numeric_score.idxmin()
            else:
                best_idx = df.index[0]

            best_row = df.loc[best_idx]

            def pick_value(candidates: list[str]) -> float:
                for c in candidates:
                    if c in df.columns:
                        return float(pd.to_numeric(best_row[c], errors="coerce"))
                return float("nan")

            result["autots_score"] = pick_value(["Validation Score", "validation_score", "Score", "score"])
            result["validation_smape"] = pick_value(["smape", "SMAPE", "Validation SMAPE", "smape_weighted"])
            result["validation_mae"] = pick_value(["mae", "MAE", "Validation MAE", "mae_weighted"])
            result["validation_rmse"] = pick_value(["rmse", "RMSE", "Validation RMSE", "rmse_weighted"])

            model_name_cols = ["Model", "model_name", "ID", "ModelParameters"]
            for c in model_name_cols:
                if c in df.columns:
                    result["best_model"] = str(best_row[c])
                    break
            return result
        except Exception:
            continue

    return result


def compute_holdout_metrics(actual: pd.Series, pred: pd.Series, train_series: pd.Series) -> dict[str, float]:
    eps = 1e-9
    y_true = actual.astype(float)
    y_pred = pred.astype(float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)
    smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / np.maximum(np.abs(y_true) + np.abs(y_pred), eps)) * 100.0)

    naive_denom = float(np.mean(np.abs(np.diff(train_series.astype(float).values))))
    if naive_denom <= eps:
        mase = float("nan")
    else:
        mase = mae / naive_denom

    return {
        "holdout_mae": mae,
        "holdout_rmse": rmse,
        "holdout_mape_pct": mape,
        "holdout_smape_pct": smape,
        "holdout_mase": float(mase),
    }


def run_holdout_backtest(
    train_df: pd.DataFrame,
    logger: Logger,
) -> HoldoutBacktestResult:
    n = len(train_df)
    holdout_len = min(60, max(14, n // 5))
    if n <= holdout_len + 30:
        logger.info("Not enough history for holdout backtest; skipping extra metrics.")
        return HoldoutBacktestResult(
            metrics={
                "holdout_mae": float("nan"),
                "holdout_rmse": float("nan"),
                "holdout_mape_pct": float("nan"),
                "holdout_smape_pct": float("nan"),
                "holdout_mase": float("nan"),
                "holdout_length": float("nan"),
            },
            holdout_length=holdout_len,
        )

    train_part = train_df.iloc[:-holdout_len].copy()
    test_part = train_df.iloc[-holdout_len:].copy()

    train_target = train_part[[TARGET_COLUMN]]
    fit_fr = train_part[FIT_REGRESSOR_COLUMNS]
    pred_fr = test_part[FIT_REGRESSOR_COLUMNS]

    logger.info(f"Running holdout backtest with holdout_length={holdout_len} days.")

    backtest_model = AutoTS(
        forecast_length=holdout_len,
        frequency=FREQUENCY,
        prediction_interval=PREDICTION_INTERVAL,
        max_generations=max(3, min(6, MAX_GENERATIONS)),
        num_validations=1,
        validation_method=VALIDATION_METHOD,
        ensemble=ENSEMBLE,
        no_negatives=NO_NEGATIVES,
    )

    try:
        backtest_model = backtest_model.fit(train_target, future_regressor=fit_fr)
        bt_pred = backtest_model.predict(future_regressor=pred_fr)
        bt_fcst = bt_pred.forecast.copy()
    except Exception as exc:
        logger.info(f"Holdout backtest failed; metrics set to NaN. reason={exc}")
        return HoldoutBacktestResult(
            metrics={
                "holdout_mae": float("nan"),
                "holdout_rmse": float("nan"),
                "holdout_mape_pct": float("nan"),
                "holdout_smape_pct": float("nan"),
                "holdout_mase": float("nan"),
                "holdout_length": float(holdout_len),
            },
            holdout_length=holdout_len,
        )

    if TARGET_COLUMN not in bt_fcst.columns and len(bt_fcst.columns) > 0:
        bt_fcst = bt_fcst.rename(columns={bt_fcst.columns[0]: TARGET_COLUMN})

    bt_fcst = bt_fcst.reindex(test_part.index)[TARGET_COLUMN]
    metrics = compute_holdout_metrics(
        actual=test_part[TARGET_COLUMN],
        pred=bt_fcst,
        train_series=train_part[TARGET_COLUMN],
    )
    metrics["holdout_length"] = float(holdout_len)
    return HoldoutBacktestResult(
        metrics=metrics,
        holdout_length=holdout_len,
        actual=test_part[TARGET_COLUMN].copy(),
        forecast=bt_fcst.copy(),
        lastvalue=pd.Series(
            np.repeat(float(train_part[TARGET_COLUMN].iloc[-1]), len(test_part)),
            index=test_part.index,
            name=TARGET_COLUMN,
        ),
    )


def make_artifact_paths(output_root_dir: Path, script_path: Path) -> RunArtifacts:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = output_root_dir / f"{script_path.stem}_{timestamp}"
    return RunArtifacts(
        timestamp=timestamp,
        run_output_dir=run_output_dir,
        forecast_csv=run_output_dir / f"forecast_365d_{timestamp}.csv",
        metrics_csv=run_output_dir / f"model_metrics_{timestamp}.csv",
        log_txt=run_output_dir / f"training_log_{timestamp}.txt",
        model_pickle=run_output_dir / f"autots_model_{timestamp}.pkl",
    )


def main() -> None:
    logger = Logger()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = make_artifact_paths(OUTPUT_DIR, Path(__file__))
    artifacts.run_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting AutoTS 365-day forecast pipeline.")
    logger.info(f"Train file: {TRAIN_CSV}")
    logger.info(f"TMY file: {TMY_CSV}")
    logger.info(f"Run output dir: {artifacts.run_output_dir}")

    try:
        train_df = read_train_data(TRAIN_CSV, logger)
        tmy_df = read_tmy_data(TMY_CSV, logger)

        train_target = train_df[[TARGET_COLUMN]]
        fit_fr = train_df[FIT_REGRESSOR_COLUMNS]

        forecast_start = train_df.index.max() + pd.Timedelta(days=1)
        forecast_index = pd.date_range(forecast_start, periods=FORECAST_LENGTH, freq=FREQUENCY)
        pred_fr = build_predict_regressor(tmy_df=tmy_df, forecast_index=forecast_index, logger=logger)

        logger.info(
            f"Forecast horizon: {forecast_index.min().date()} -> {forecast_index.max().date()} "
            f"({FORECAST_LENGTH} days)"
        )

        fit_forecast_length = min(FORECAST_LENGTH, max(30, len(train_target) // 3))

        model = AutoTS(
            forecast_length=fit_forecast_length,
            frequency=FREQUENCY,
            prediction_interval=PREDICTION_INTERVAL,
            max_generations=MAX_GENERATIONS,
            num_validations=NUM_VALIDATIONS,
            validation_method=VALIDATION_METHOD,
            ensemble=ENSEMBLE,
            no_negatives=NO_NEGATIVES,
        )

        logger.info(
            "Fitting AutoTS model with future_regressor on training data. "
            f"fit_forecast_length={fit_forecast_length}, predict_forecast_length={FORECAST_LENGTH}"
        )

        try:
            model = model.fit(train_target, future_regressor=fit_fr)
        except ValueError as exc:
            msg = str(exc).lower()
            if "forecast_length is too large" in msg:
                logger.info(
                    "Fit failed due to forecast_length/CV constraint; retrying with num_validations=0."
                )
                model = AutoTS(
                    forecast_length=fit_forecast_length,
                    frequency=FREQUENCY,
                    prediction_interval=PREDICTION_INTERVAL,
                    max_generations=MAX_GENERATIONS,
                    num_validations=0,
                    validation_method=VALIDATION_METHOD,
                    ensemble=ENSEMBLE,
                    no_negatives=NO_NEGATIVES,
                )
                model = model.fit(train_target, future_regressor=fit_fr)
            else:
                raise

        logger.info("Predicting 365 days with TMY-based future_regressor.")
        try:
            prediction = model.predict(forecast_length=FORECAST_LENGTH, future_regressor=pred_fr)
        except Exception as exc:
            msg = str(exc).lower()
            if "out of bounds for int8" in msg or "overflow" in msg:
                logger.info(
                    "Predict failed on ensemble='all' long horizon; "
                    "retrying with stable fallback ensemble='simple'."
                )
                fallback_model = AutoTS(
                    forecast_length=fit_forecast_length,
                    frequency=FREQUENCY,
                    prediction_interval=PREDICTION_INTERVAL,
                    max_generations=max(6, min(10, MAX_GENERATIONS)),
                    num_validations=0,
                    validation_method=VALIDATION_METHOD,
                    ensemble="simple",
                    model_list="fast",
                    no_negatives=NO_NEGATIVES,
                )
                fallback_model = fallback_model.fit(train_target, future_regressor=fit_fr)
                prediction = fallback_model.predict(
                    forecast_length=FORECAST_LENGTH,
                    future_regressor=pred_fr,
                )
                model = fallback_model
            else:
                raise

        fcst = prediction.forecast.copy()
        lower = prediction.lower_forecast.copy()
        upper = prediction.upper_forecast.copy()

        if TARGET_COLUMN not in fcst.columns and len(fcst.columns) > 0:
            src = fcst.columns[0]
            fcst = fcst.rename(columns={src: TARGET_COLUMN})
            lower = lower.rename(columns={lower.columns[0]: TARGET_COLUMN})
            upper = upper.rename(columns={upper.columns[0]: TARGET_COLUMN})

        out_forecast = pd.DataFrame(
            {
                "date": forecast_index,
                "forecast": fcst.reindex(forecast_index)[TARGET_COLUMN].values,
                "lower_bound": lower.reindex(forecast_index)[TARGET_COLUMN].values,
                "upper_bound": upper.reindex(forecast_index)[TARGET_COLUMN].values,
            }
        )

        out_forecast.to_csv(artifacts.forecast_csv, index=False, encoding="utf-8-sig")
        logger.info(f"Saved forecast CSV: {artifacts.forecast_csv}")

        validation_metrics = extract_best_validation_metrics(model)
        holdout_result = run_holdout_backtest(train_df=train_df, logger=logger)
        holdout_metrics = holdout_result.metrics

        if holdout_result.actual is not None and holdout_result.forecast is not None and holdout_result.lastvalue is not None:
            holdout_horizon = holdout_result.holdout_length
            comparison_title = f"{TARGET_COLUMN} Forecast vs Actual vs LastValueNaive"
            format2_plot = artifacts.run_output_dir / f"AutoTS_forecast_vs_actual_vs_lastvalue_{holdout_horizon}-format2.png"
            legacy_plot = artifacts.run_output_dir / f"forecast_vs_actual_vs_lastvalue_{holdout_horizon}.png"

            if save_forecast_vs_actual_vs_lastvalue_plot(
                plot_path=format2_plot,
                actual=holdout_result.actual,
                forecast=holdout_result.forecast,
                lastvalue=holdout_result.lastvalue,
                title=comparison_title,
                logger=logger,
                metrics=holdout_metrics,
            ):
                logger.info(f"Saved holdout comparison plot: {format2_plot}")

            if save_forecast_vs_actual_vs_lastvalue_plot(
                plot_path=legacy_plot,
                actual=holdout_result.actual,
                forecast=holdout_result.forecast,
                lastvalue=holdout_result.lastvalue,
                title=comparison_title,
                logger=logger,
                metrics=holdout_metrics,
            ):
                logger.info(f"Saved holdout comparison plot: {legacy_plot}")

        future_forecast_plot = artifacts.run_output_dir / f"forecast_365d_future_{artifacts.timestamp}.png"
        future_forecast_values = fcst.reindex(forecast_index)[TARGET_COLUMN].values
        future_lower_values = lower.reindex(forecast_index)[TARGET_COLUMN].values
        future_upper_values = upper.reindex(forecast_index)[TARGET_COLUMN].values
        if save_future_forecast_plot(
            plot_path=future_forecast_plot,
            forecast_index=forecast_index,
            forecast_values=future_forecast_values,
            lower_values=future_lower_values,
            upper_values=future_upper_values,
            title=f"{TARGET_COLUMN} 365-Day Future Forecast",
            logger=logger,
        ):
            logger.info(f"Saved 365-day future forecast plot: {future_forecast_plot}")

        metrics_payload = {
            "timestamp": artifacts.timestamp,
            "train_file": str(TRAIN_CSV),
            "tmy_file": str(TMY_CSV),
            "target_column": TARGET_COLUMN,
            "fit_regressors": ",".join(FIT_REGRESSOR_COLUMNS),
            "train_rows": len(train_df),
            "forecast_length": FORECAST_LENGTH,
            "forecast_start": str(forecast_index.min().date()),
            "forecast_end": str(forecast_index.max().date()),
            "frequency": FREQUENCY,
            "prediction_interval": PREDICTION_INTERVAL,
            "max_generations": MAX_GENERATIONS,
            "num_validations": NUM_VALIDATIONS,
            "validation_method": VALIDATION_METHOD,
            "ensemble": ENSEMBLE,
            "no_negatives": NO_NEGATIVES,
            "best_model": validation_metrics.get("best_model", ""),
            "autots_score": validation_metrics.get("autots_score", np.nan),
            "validation_smape": validation_metrics.get("validation_smape", np.nan),
            "validation_mae": validation_metrics.get("validation_mae", np.nan),
            "validation_rmse": validation_metrics.get("validation_rmse", np.nan),
            "holdout_mae": holdout_metrics.get("holdout_mae", np.nan),
            "holdout_rmse": holdout_metrics.get("holdout_rmse", np.nan),
            "holdout_mape_pct": holdout_metrics.get("holdout_mape_pct", np.nan),
            "holdout_mase": holdout_metrics.get("holdout_mase", np.nan),
            "holdout_length": holdout_metrics.get("holdout_length", np.nan),
        }
        pd.DataFrame([metrics_payload]).to_csv(artifacts.metrics_csv, index=False, encoding="utf-8-sig")
        logger.info(f"Saved metrics CSV: {artifacts.metrics_csv}")

        with artifacts.model_pickle.open("wb") as f:
            pickle.dump(model, f)
        logger.info(f"Saved model pickle: {artifacts.model_pickle}")

        logger.info("Run completed successfully.")
    except Exception as exc:
        logger.info(f"Run failed: {exc}")
        logger.info(traceback.format_exc())
        raise
    finally:
        logger.dump(artifacts.log_txt)
        print(f"Training log saved to: {artifacts.log_txt}")


if __name__ == "__main__":
    main()
