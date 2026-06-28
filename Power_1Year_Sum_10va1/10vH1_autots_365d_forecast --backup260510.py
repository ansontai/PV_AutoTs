from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
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


# FIT_FORECAST_LENGTH = "auto"
FIT_FORECAST_LENGTH = 120
FORECAST_LENGTH = 365
FREQUENCY = "D"
PREDICTION_INTERVAL = 0.9
# MAX_GENERATIONS = 15
# MAX_GENERATIONS = 30
MAX_GENERATIONS = 50
NUM_VALIDATIONS = 3
VALIDATION_METHOD = "backwards"
ENSEMBLE = "all"
NO_NEGATIVES = True

TARGET_COLUMN = "Wh"
DATE_CANDIDATES = ("date", "Date", "LocalTime")
PHYSICAL_REGRESSORS = ["Temperature", "RH", "GloblRad"]
ALL_REGRESSORS = [
    "Temperature", "RH", "GloblRad", "day_of_year",
    "sin_DOY_k1", "cos_DOY_k1", "sin_DOY_k2", "cos_DOY_k2",
    "sin_DOY_k3", "cos_DOY_k3",
    "month_2", "month_3", "month_4", "month_5", "month_6",
    "month_7", "month_8", "month_9", "month_10", "month_11", "month_12",
    "season_spring", "season_summer", "season_autumn",
    "is_weekend"
]
FIT_REGRESSOR_COLUMNS = PHYSICAL_REGRESSORS

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
TRAIN_CSV = INPUT_DIR / "SolarRecord(260228)_d_forWh_WithCodis[date].csv"
TMY_CSV = INPUT_DIR / "tmy_24.148_120.703_2005_2023[UTC+8][daily][mapped][dateAdj].csv"
PVGIS_TIMESERIES_CSV = INPUT_DIR / "Timeseries_24.148_120.703_E5_0kWp_crystSi_25_35deg_1deg_2005_2005[UTC+8][daily][scaled][dateAdj].csv"
PVGIS_COLUMN = "P_mapped_Wh"

MODEL_LIST_FALLBACK_SAFE = [
    "ConstantNaive",
    "LastValueNaive",
    "AverageValueNaive",
    "SeasonalNaive",
    "GLS",
    "GLM",
    "ETS",
    "Theta",
]

ENSEMBLE_FALLBACK_SAFE = "simple"


@dataclass
class RunArtifacts:
    timestamp: str
    run_output_dir: Path
    forecast_csv: Path
    templates_csv: Path
    templates_json: Path
    best_template_csv: Path
    best_template_json: Path
    metrics_csv: Path
    settings_csv: Path
    settings_json: Path
    weights_csv: Path
    weights_json: Path
    log_txt: Path
    model_pickle: Path
    consistency_csv: Path


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


def read_pvgis_timeseries_data(csv_path: Path, logger: Logger) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"PVGIS timeseries CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    date_col = detect_date_column(df.columns)
    if PVGIS_COLUMN not in df.columns:
        raise ValueError(f"PVGIS timeseries CSV missing required column: {PVGIS_COLUMN}")

    out = df[[date_col, PVGIS_COLUMN]].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[PVGIS_COLUMN] = pd.to_numeric(out[PVGIS_COLUMN], errors="coerce")

    out = out.dropna(subset=[date_col]).sort_values(date_col)
    out = out.drop_duplicates(subset=[date_col], keep="last")
    out = out.rename(columns={date_col: "date"}).set_index("date")
    out = out.asfreq(FREQUENCY)
    out[PVGIS_COLUMN] = out[PVGIS_COLUMN].interpolate(limit_direction="both")

    logger.info(
        "Loaded PVGIS timeseries data: "
        f"rows={len(out)}, date_range={out.index.min().date()} -> {out.index.max().date()}"
    )
    return out


def build_pvgis_series(
    pvgis_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    logger: Logger,
) -> pd.Series:
    logger.info("Building PVGIS series by month-day mapping (year is ignored).")
    pvgis_tmp = pvgis_df[[PVGIS_COLUMN]].copy()
    pvgis_tmp["md"] = pvgis_tmp.index.strftime("%m-%d")
    by_md = pvgis_tmp.groupby("md")[[PVGIS_COLUMN]].mean()

    md_keys = target_index.strftime("%m-%d")
    out = by_md.reindex(md_keys)[PVGIS_COLUMN]
    out.index = target_index

    if out.isna().any():
        logger.info("PVGIS month-day mapping has missing values; applying ffill/bfill fallback.")

    out = out.ffill().bfill()
    if out.isna().any():
        raise ValueError("PVGIS P_mapped_Wh still has NaN after fallback fill.")

    out.name = PVGIS_COLUMN
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


def save_pvgis_forecast_vs_actual_vs_lastvalue_plot(
    plot_path: Path,
    actual: pd.Series,
    pvgis_forecast: pd.Series,
    lastvalue: pd.Series,
    logger: Logger,
) -> bool:
    if not MATPLOTLIB_AVAILABLE:
        logger.info(f"Matplotlib not available; skipping plot: {plot_path}")
        return False

    actual = actual.astype(float)
    pvgis_forecast = pvgis_forecast.astype(float)
    lastvalue = lastvalue.astype(float)

    fig, ax = plt.subplots(figsize=(6, 3), dpi=300, constrained_layout=True)
    ax.plot(actual.index, actual.values, label="Actual", color="black", linewidth=2.5)
    ax.plot(pvgis_forecast.index, pvgis_forecast.values, label="PVGIS Forecast", color="dimgray", linewidth=2.5)
    ax.plot(lastvalue.index, lastvalue.values, label="LastValueNaive", color="gray", linewidth=2, linestyle="--")
    ax.set_title(f"{PVGIS_COLUMN} vs Actual vs LastValueNaive", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel(TARGET_COLUMN)
    ax.grid(alpha=0.35, linestyle=":", linewidth=0.8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 0.60), fontsize=9, frameon=False)
    fig.subplots_adjust(bottom=0.18, right=0.80)
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
    forecast_length: int | None = None,
) -> HoldoutBacktestResult:
    """Run a holdout backtest.

    If `forecast_length` is provided it will be used as the holdout horizon; otherwise
    the legacy adaptive `holdout_len = min(60, max(14, n//5))` is used.
    """
    n = len(train_df)
    if forecast_length is None:
        holdout_len = min(60, max(14, n // 5))
        mode = "auto"
    else:
        try:
            holdout_len = int(forecast_length)
            mode = "fixed"
        except Exception:
            logger.info(f"Invalid forecast_length provided for holdout backtest: {forecast_length}; falling back to auto.")
            holdout_len = min(60, max(14, n // 5))
            mode = "auto"

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

    logger.info(f"Running holdout backtest with holdout_length={holdout_len} days (mode={mode}).")

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
        templates_csv=run_output_dir / f"autots_templates_{timestamp}.csv",
        templates_json=run_output_dir / f"autots_templates_{timestamp}.json",
        best_template_csv=run_output_dir / f"autots_best_template_{timestamp}.csv",
        best_template_json=run_output_dir / f"autots_best_template_{timestamp}.json",
        weights_csv=run_output_dir / f"weight_summary_{timestamp}.csv",
        weights_json=run_output_dir / f"weight_summary_{timestamp}.json",
        metrics_csv=run_output_dir / f"model_metrics_{timestamp}.csv",
        settings_csv=run_output_dir / f"effective_settings_{timestamp}.csv",
        settings_json=run_output_dir / f"effective_settings_{timestamp}.json",
        log_txt=run_output_dir / f"training_log_{timestamp}.txt",
        model_pickle=run_output_dir / f"autots_model_{timestamp}.pkl",
        consistency_csv=run_output_dir / f"forecast_365d_consistency_{timestamp}.csv",
    )


def save_effective_settings_csv(csv_path: Path, payload: dict[str, object], logger: Logger) -> None:
    pd.DataFrame([payload]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved effective settings CSV: {csv_path}")


def save_effective_settings_json(json_path: Path, payload: dict[str, object], logger: Logger) -> None:
    json_path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved effective settings JSON: {json_path}")


def _json_safe(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    if value is None:
        return None
    return value


def _maybe_parse_json(value):
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                return json.loads(text)
            except Exception:
                return value
    return value


def _first_existing_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _extract_results_frame(model: AutoTS) -> pd.DataFrame | None:
    for result_set in ("validation", None):
        try:
            df = model.results(result_set) if result_set else model.results()
        except Exception:
            continue
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.copy()
    return None


def _extract_best_template_row(model: AutoTS) -> dict[str, object] | None:
    df = _extract_results_frame(model)
    if df is None or df.empty:
        return None

    score_col = _first_existing_column(
        df.columns,
        ["Validation Score", "validation_score", "Score", "score", "smape_weighted"],
    )
    ordered = df.copy()
    if score_col is not None:
        score_series = pd.to_numeric(ordered[score_col], errors="coerce")
        if score_series.notna().any():
            ordered = ordered.assign(_score_numeric=score_series)
            ordered = ordered.sort_values(by=["_score_numeric"], kind="mergesort", na_position="last")
        else:
            ordered = ordered.reset_index(drop=True)
    else:
        ordered = ordered.reset_index(drop=True)

    best_row = ordered.iloc[0].to_dict()
    best_row = {str(key): _json_safe(value) for key, value in best_row.items() if not str(key).startswith("_")}
    best_row["source"] = "model.results"
    return best_row


def _extract_model_weight_rows(model: AutoTS) -> tuple[list[dict[str, object]], dict[str, object]]:
    df = _extract_results_frame(model)
    meta = {
        "score_column": None,
        "weight_method": "unavailable",
        "row_count": 0,
    }
    if df is None or df.empty:
        return [], meta

    score_col = _first_existing_column(
        df.columns,
        ["Validation Score", "validation_score", "Score", "score", "smape_weighted"],
    )
    model_col = _first_existing_column(df.columns, ["Model", "model_name", "ID"])
    param_col = _first_existing_column(df.columns, ["ModelParameters", "model_params"])
    trans_col = _first_existing_column(df.columns, ["TransformationParameters", "transformation_params"])
    ensemble_col = _first_existing_column(df.columns, ["Ensemble", "ensemble"])
    dedup_cols = [c for c in [model_col, param_col, trans_col, ensemble_col] if c is not None]
    if dedup_cols:
        df = df.sort_values(by=score_col if score_col in df.columns else df.columns[0], kind="mergesort", na_position="last")
        dedup_frame = pd.DataFrame(index=df.index)
        for column in dedup_cols:
            dedup_frame[column] = df[column].map(lambda value: json.dumps(_json_safe(value), sort_keys=True, ensure_ascii=False))
        df = df.loc[~dedup_frame.duplicated(keep="first")].copy()

    score_series = pd.to_numeric(df[score_col], errors="coerce") if score_col is not None else pd.Series(dtype=float)
    has_numeric_score = bool(score_series.notna().any())
    if has_numeric_score:
        ordered = df.assign(_score_numeric=score_series)
        ordered = ordered.sort_values(by=["_score_numeric"], kind="mergesort", na_position="last").reset_index(drop=True)
        meta["score_column"] = score_col
        meta["weight_method"] = "inverse_rank_from_validation_score"
    else:
        ordered = df.reset_index(drop=True).copy()
        meta["weight_method"] = "inverse_rank_from_result_order"

    if len(ordered) == 0:
        return [], meta

    ranks = np.arange(1, len(ordered) + 1, dtype=float)
    rank_weights = 1.0 / ranks
    rank_weights = rank_weights / rank_weights.sum()
    ordered["_rank"] = np.arange(1, len(ordered) + 1, dtype=int)
    ordered["_usage_weight"] = rank_weights
    meta["row_count"] = int(len(ordered))

    rows: list[dict[str, object]] = []
    for _, row in ordered.iterrows():
        score_value = None
        if score_col is not None and score_col in ordered.columns:
            score_value = _json_safe(row.get(score_col))
        rows.append(
            {
                "group": "model_weights",
                "rank": int(row["_rank"]),
                "usage_weight": float(row["_usage_weight"]),
                "weight_method": meta["weight_method"],
                "score_column": score_col or "",
                "score_value": score_value,
                "model_name": _json_safe(row.get(model_col)) if model_col else None,
                "model_parameters": _json_safe(_maybe_parse_json(row.get(param_col))) if param_col else None,
                "transformation_parameters": _json_safe(_maybe_parse_json(row.get(trans_col))) if trans_col else None,
                "ensemble": _json_safe(row.get(ensemble_col)) if ensemble_col else None,
                "source": "model.results",
                "is_best": bool(int(row["_rank"]) == 1),
            }
        )

    return rows, meta


def _extract_feature_weight_rows(train_df: pd.DataFrame) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    meta = {
        "weight_method": "abs_pearson_correlation_on_training_set",
        "row_count": 0,
    }

    rows.append(
        {
            "group": "feature_weights",
            "feature_name": TARGET_COLUMN,
            "usage_weight": 1.0,
            "raw_score": 1.0,
            "weight_method": "fixed_target",
            "source": "training_target",
            "is_target": True,
        }
    )

    regressor_rows: list[dict[str, object]] = []
    for column in FIT_REGRESSOR_COLUMNS:
        raw_score = float("nan")
        try:
            pair = train_df[[TARGET_COLUMN, column]].dropna()
            if len(pair) >= 2 and pair[TARGET_COLUMN].nunique(dropna=True) > 1 and pair[column].nunique(dropna=True) > 1:
                raw_score = abs(float(pair[TARGET_COLUMN].corr(pair[column])))
        except Exception:
            raw_score = float("nan")
        regressor_rows.append(
            {
                "group": "feature_weights",
                "feature_name": column,
                "usage_weight": 0.0,
                "raw_score": raw_score,
                "weight_method": meta["weight_method"],
                "source": "future_regressor",
                "is_target": False,
            }
        )

    valid_rows = [row for row in regressor_rows if isinstance(row["raw_score"], (int, float, np.floating)) and not np.isnan(row["raw_score"])]
    if valid_rows:
        total_score = float(sum(float(row["raw_score"]) for row in valid_rows))
        if total_score > 0:
            for row in valid_rows:
                row["usage_weight"] = float(row["raw_score"]) / total_score
        else:
            equal_weight = 1.0 / len(valid_rows)
            for row in valid_rows:
                row["usage_weight"] = equal_weight
    elif regressor_rows:
        equal_weight = 1.0 / len(regressor_rows)
        for row in regressor_rows:
            row["usage_weight"] = equal_weight

    rows.extend(regressor_rows)
    meta["row_count"] = int(len(rows))
    return rows, meta


def _flatten_selected_parameter_rows(
    rows: list[dict[str, object]],
    model_name: str,
    ensemble_value,
    section: str,
    value,
    path_prefix: str,
) -> None:
    parsed_value = _maybe_parse_json(value)
    if isinstance(parsed_value, dict):
        if not parsed_value:
            rows.append(
                {
                    "group": "parameter_weights",
                    "parameter_group": section,
                    "parameter_path": path_prefix,
                    "parameter_value": {},
                    "usage_weight": 1.0,
                    "weight_method": "selected_in_best_model",
                    "model_name": model_name,
                    "ensemble": _json_safe(ensemble_value),
                    "source": "best_model",
                }
            )
        else:
            for key, child_value in parsed_value.items():
                child_path = f"{path_prefix}.{key}" if path_prefix else str(key)
                _flatten_selected_parameter_rows(rows, model_name, ensemble_value, section, child_value, child_path)
        return

    if isinstance(parsed_value, (list, tuple)):
        if not parsed_value:
            rows.append(
                {
                    "group": "parameter_weights",
                    "parameter_group": section,
                    "parameter_path": path_prefix,
                    "parameter_value": [],
                    "usage_weight": 1.0,
                    "weight_method": "selected_in_best_model",
                    "model_name": model_name,
                    "ensemble": _json_safe(ensemble_value),
                    "source": "best_model",
                }
            )
        else:
            for index, child_value in enumerate(parsed_value):
                child_path = f"{path_prefix}[{index}]"
                _flatten_selected_parameter_rows(rows, model_name, ensemble_value, section, child_value, child_path)
        return

    rows.append(
        {
            "group": "parameter_weights",
            "parameter_group": section,
            "parameter_path": path_prefix,
            "parameter_value": _json_safe(parsed_value),
            "usage_weight": 1.0,
            "weight_method": "selected_in_best_model",
            "model_name": model_name,
            "ensemble": _json_safe(ensemble_value),
            "source": "best_model",
        }
    )


def _extract_parameter_weight_rows(model: AutoTS) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    best_model_df = getattr(model, "best_model", None)
    best_model_name = getattr(model, "best_model_name", None)
    best_model_params = getattr(model, "best_model_params", None)
    best_model_transformation_params = getattr(model, "best_model_transformation_params", None)
    best_model_ensemble = getattr(model, "best_model_ensemble", None)

    if isinstance(best_model_df, pd.DataFrame) and not best_model_df.empty:
        best_row = best_model_df.iloc[0].to_dict()
        best_model_name = best_model_name or best_row.get("Model") or best_row.get("model")
        best_model_params = best_model_params or best_row.get("ModelParameters") or best_row.get("model_params")
        best_model_transformation_params = best_model_transformation_params or best_row.get("TransformationParameters") or best_row.get("transformation_params")
        if best_model_ensemble is None:
            best_model_ensemble = best_row.get("Ensemble") or best_row.get("ensemble")

    summary = {
        "best_model_name": _json_safe(best_model_name),
        "best_model_ensemble": _json_safe(best_model_ensemble),
        "best_model_params": _json_safe(_maybe_parse_json(best_model_params)),
        "best_model_transformation_params": _json_safe(_maybe_parse_json(best_model_transformation_params)),
    }

    _flatten_selected_parameter_rows(
        rows=rows,
        model_name=str(best_model_name) if best_model_name is not None else "",
        ensemble_value=best_model_ensemble,
        section="ModelParameters",
        value=best_model_params,
        path_prefix="ModelParameters",
    )
    _flatten_selected_parameter_rows(
        rows=rows,
        model_name=str(best_model_name) if best_model_name is not None else "",
        ensemble_value=best_model_ensemble,
        section="TransformationParameters",
        value=best_model_transformation_params,
        path_prefix="TransformationParameters",
    )

    if not rows:
        rows.append(
            {
                "group": "parameter_weights",
                "parameter_group": "best_model",
                "parameter_path": "best_model_name",
                "parameter_value": _json_safe(best_model_name),
                "usage_weight": 1.0,
                "weight_method": "selected_in_best_model",
                "model_name": _json_safe(best_model_name),
                "ensemble": _json_safe(best_model_ensemble),
                "source": "best_model",
            }
        )

    meta = {"row_count": int(len(rows)), "weight_method": "selected_in_best_model"}
    return rows, {"summary": summary, **meta}


def save_template_artifacts(csv_path: Path, json_path: Path, model: AutoTS, logger: Logger) -> dict[str, object]:
    templates_df = _extract_results_frame(model)
    if templates_df is None or templates_df.empty:
        logger.info("No AutoTS templates/results frame available; skipping template artifact save.")
        return {"row_count": 0, "best_template": None}

    templates_df = templates_df.copy()
    templates_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps(_json_safe(templates_df.to_dict(orient="records")), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_template = _extract_best_template_row(model)
    row_count = int(len(templates_df))
    logger.info(f"Saved AutoTS templates CSV: {csv_path}")
    logger.info(f"Saved AutoTS templates JSON: {json_path}")
    return {"row_count": row_count, "best_template": best_template}


def save_best_template_artifacts(csv_path: Path, json_path: Path, best_template: dict[str, object] | None, logger: Logger) -> None:
    if not best_template:
        logger.info("No best template summary available; skipping best template artifact save.")
        return

    pd.DataFrame([best_template]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(_json_safe(best_template), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved AutoTS best template CSV: {csv_path}")
    logger.info(f"Saved AutoTS best template JSON: {json_path}")


def build_weight_artifacts(model: AutoTS, train_df: pd.DataFrame) -> dict[str, object]:
    model_rows, model_meta = _extract_model_weight_rows(model)
    feature_rows, feature_meta = _extract_feature_weight_rows(train_df)
    parameter_rows, parameter_meta = _extract_parameter_weight_rows(model)

    csv_rows = model_rows + feature_rows + parameter_rows
    payload = {
        "model_weights": model_rows,
        "feature_weights": feature_rows,
        "parameter_weights": parameter_rows,
        "best_model_summary": parameter_meta.get("summary", {}),
        "weight_methods": {
            "model_weights": model_meta.get("weight_method"),
            "feature_weights": feature_meta.get("weight_method"),
            "parameter_weights": parameter_meta.get("weight_method"),
        },
        "row_counts": {
            "model_weights": model_meta.get("row_count", 0),
            "feature_weights": feature_meta.get("row_count", 0),
            "parameter_weights": parameter_meta.get("row_count", 0),
        },
    }

    return {
        "payload": payload,
        "csv_rows": csv_rows,
        "model_weight_count": int(model_meta.get("row_count", 0)),
        "feature_weight_count": int(feature_meta.get("row_count", 0)),
        "parameter_weight_count": int(parameter_meta.get("row_count", 0)),
        "best_model_name": parameter_meta.get("summary", {}).get("best_model_name"),
        "best_model_ensemble": parameter_meta.get("summary", {}).get("best_model_ensemble"),
    }


def save_weight_artifacts(csv_path: Path, json_path: Path, weight_artifacts: dict[str, object], logger: Logger) -> None:
    csv_rows = weight_artifacts.get("csv_rows", [])
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(_json_safe(weight_artifacts.get("payload", {})), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved weight summary CSV: {csv_path}")
    logger.info(f"Saved weight summary JSON: {json_path}")


def compute_forecast_aggregates(forecast_df: pd.DataFrame, freq: str = "A") -> pd.DataFrame:
    """Aggregate daily forecast rows into periods defined by `freq`.

    Returns a DataFrame with columns: period_start, total_forecast, total_lower, total_upper
    """
    if forecast_df is None or len(forecast_df) == 0:
        return pd.DataFrame(columns=["period_start", "total_forecast", "total_lower", "total_upper"])

    df = forecast_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    agg = df[["forecast", "lower_bound", "upper_bound"]].resample(freq).sum()
    agg = agg.rename(columns={"forecast": "total_forecast", "lower_bound": "total_lower", "upper_bound": "total_upper"})
    agg = agg.reset_index()
    agg = agg.rename(columns={"date": "period_start"})
    return agg[["period_start", "total_forecast", "total_lower", "total_upper"]]


def save_aggregated_forecasts(aggregates_df: pd.DataFrame, out_dir: Path, prefix: str, timestamp: str, logger: Logger) -> Path:
    """Save aggregated forecast DataFrame to CSV and return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"forecast_365d_{prefix}_{timestamp}.csv"
    aggregates_df.to_csv(file_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved {prefix} aggregated forecast CSV: {file_path}")
    return file_path


def save_forecast_totals(out_dir: Path, timestamp: str, forecast_df: pd.DataFrame, logger: Logger) -> Path:
    """Save one-row totals for the forecast output columns."""
    out_dir.mkdir(parents=True, exist_ok=True)
    totals_df = pd.DataFrame(
        [
            {
                "total_forecast": float(pd.to_numeric(forecast_df["forecast"], errors="coerce").sum()),
                "total_lower": float(pd.to_numeric(forecast_df["lower_bound"], errors="coerce").sum()),
                "total_upper": float(pd.to_numeric(forecast_df["upper_bound"], errors="coerce").sum()),
                "total_P_mapped_Wh": float(pd.to_numeric(forecast_df[PVGIS_COLUMN], errors="coerce").sum()),
            }
        ]
    )
    file_path = out_dir / f"forecast_365d_totals_{timestamp}.csv"
    totals_df.to_csv(file_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved forecast totals CSV: {file_path}")
    return file_path


def compute_and_save_annual_consistency(totals_csv_path: Path, output_dir: Path, timestamp: str, logger: Logger) -> Path:
    """Compute annual consistency metrics (bias and relative error) and save to CSV.
    
    Annual_Bias = Predicted_Annual (total_forecast) - Reference_Annual (total_P_mapped_Wh)
    Relative_Error_RE (%) = |Pred - Reference| / Reference × 100%
    """
    if not totals_csv_path.exists():
        logger.info(f"Totals CSV not found: {totals_csv_path}; skipping consistency metrics.")
        return None
    
    try:
        totals_df = pd.read_csv(totals_csv_path)
        if totals_df.empty:
            logger.info("Totals CSV is empty; skipping consistency metrics.")
            return None
        
        row = totals_df.iloc[0]
        predicted_annual = float(pd.to_numeric(row.get("total_forecast"), errors="coerce"))
        reference_annual = float(pd.to_numeric(row.get("total_P_mapped_Wh"), errors="coerce"))
        
        if np.isnan(predicted_annual) or np.isnan(reference_annual):
            logger.info("Predicted or reference annual values are NaN; skipping consistency metrics.")
            return None
        
        if reference_annual == 0:
            logger.info("Reference annual value is 0; cannot compute relative error.")
            return None
        
        annual_bias = predicted_annual - reference_annual
        relative_error_re = abs(predicted_annual - reference_annual) / abs(reference_annual) * 100.0
        
        consistency_df = pd.DataFrame(
            [
                {
                    "Predicted_Annual_Wh": predicted_annual,
                    "Reference_Annual_Wh": reference_annual,
                    "Annual_Bias_Wh": annual_bias,
                    "Relative_Error_RE_%": relative_error_re,
                }
            ]
        )
        
        file_path = output_dir / f"forecast_365d_consistency_{timestamp}.csv"
        consistency_df.to_csv(file_path, index=False, encoding="utf-8-sig")
        logger.info(
            f"Saved annual consistency metrics CSV: {file_path} "
            f"(Bias={annual_bias:.2f} Wh, RelError={relative_error_re:.2f}%)"
        )
        return file_path
    
    except Exception as exc:
        logger.info(f"Failed to compute annual consistency metrics: {exc}")
        return None


def resolve_fit_forecast_length(raw_value, train_length: int) -> tuple[int, str]:
    """Resolve the fit forecast length from FIT_FORECAST_LENGTH.

    Numeric values are used directly. The string "auto" keeps the legacy
    adaptive behavior.
    """
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if text.lower() == "auto":
            return min(FORECAST_LENGTH, max(30, train_length // 3)), "auto"
        try:
            return int(text), "fixed"
        except ValueError as exc:
            raise ValueError(f"Unsupported FIT_FORECAST_LENGTH value: {raw_value!r}") from exc

    try:
        return int(raw_value), "fixed"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported FIT_FORECAST_LENGTH value: {raw_value!r}") from exc


def main(args) -> None:
    global FIT_REGRESSOR_COLUMNS
    logger = Logger()
    
    # 根據開關設置特徵列表
    if args.include_temporal_features:
        FIT_REGRESSOR_COLUMNS = ALL_REGRESSORS
        logger.info(f"[CONFIG] Using ALL regressors (physical + temporal): {len(FIT_REGRESSOR_COLUMNS)} features")
    else:
        FIT_REGRESSOR_COLUMNS = PHYSICAL_REGRESSORS
        logger.info(f"[CONFIG] Using PHYSICAL regressors only: {len(FIT_REGRESSOR_COLUMNS)} features")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = make_artifact_paths(OUTPUT_DIR, Path(__file__))
    artifacts.run_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting AutoTS 365-day forecast pipeline.")
    logger.info(f"Train file: {TRAIN_CSV}")
    logger.info(f"TMY file: {TMY_CSV}")
    logger.info(f"PVGIS timeseries file: {PVGIS_TIMESERIES_CSV}")
    logger.info(f"Run output dir: {artifacts.run_output_dir}")

    annual_forecast_csv = None
    monthly_forecast_csv = None
    totals_forecast_csv = None
    consistency_forecast_csv = None

    try:
        train_df = read_train_data(TRAIN_CSV, logger)
        tmy_df = read_tmy_data(TMY_CSV, logger)
        pvgis_df = read_pvgis_timeseries_data(PVGIS_TIMESERIES_CSV, logger)

        train_target = train_df[[TARGET_COLUMN]]
        fit_fr = train_df[FIT_REGRESSOR_COLUMNS]

        forecast_start = train_df.index.max() + pd.Timedelta(days=1)
        forecast_index = pd.date_range(forecast_start, periods=FORECAST_LENGTH, freq=FREQUENCY)
        pred_fr = build_predict_regressor(tmy_df=tmy_df, forecast_index=forecast_index, logger=logger)
        pvgis_365d = build_pvgis_series(pvgis_df=pvgis_df, target_index=forecast_index, logger=logger)

        logger.info(
            f"Forecast horizon: {forecast_index.min().date()} -> {forecast_index.max().date()} "
            f"({FORECAST_LENGTH} days)"
        )

        fit_forecast_length, fit_forecast_length_mode = resolve_fit_forecast_length(
            FIT_FORECAST_LENGTH,
            len(train_target),
        )

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
            f"fit_forecast_length={fit_forecast_length} (mode={fit_forecast_length_mode}, raw={FIT_FORECAST_LENGTH!r}), "
            f"predict_forecast_length={FORECAST_LENGTH}"
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
        # Debug prints to help diagnose sliding-window / mosaic ensemble issues
        try:
            logger.info(f"DEBUG: train_target rows={len(train_target)} fit_fr_shape={getattr(fit_fr, 'shape', None)}")
            logger.info(f"DEBUG: model forecast_length param={getattr(model, 'forecast_length', None)}")
            logger.info(f"DEBUG: requested FORECAST_LENGTH={FORECAST_LENGTH}")
            logger.info(f"DEBUG: model.window={getattr(model, 'window', None)}")
            prediction = model.predict(forecast_length=FORECAST_LENGTH, future_regressor=pred_fr)
        except Exception as exc:
            msg = str(exc).lower()
            if "out of bounds for int8" in msg or "overflow" in msg:
                logger.info(
                    "Predict failed on ensemble='all' long horizon; "
                    "retrying with fallback ensemble='simple' to avoid mosaic ensemble."
                )
                fallback_model = AutoTS(
                    forecast_length=fit_forecast_length,
                    frequency=FREQUENCY,
                    prediction_interval=PREDICTION_INTERVAL,
                    max_generations=max(6, min(30, MAX_GENERATIONS)),
                    # max_generations = MAX_GENERATIONS,  # 不要降低 max_generations，讓 fallback 模型也有機會找到好模型
                    num_validations=0,
                    validation_method=VALIDATION_METHOD,
                    ensemble=ENSEMBLE_FALLBACK_SAFE,
                    model_list=MODEL_LIST_FALLBACK_SAFE,
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
                PVGIS_COLUMN: pvgis_365d.reindex(forecast_index).values,
            }
        )

        out_forecast.to_csv(artifacts.forecast_csv, index=False, encoding="utf-8-sig")
        logger.info(f"Saved forecast CSV: {artifacts.forecast_csv}")

        # Aggregate daily forecast into annual and monthly totals and save
        try:
            annual_agg = compute_forecast_aggregates(out_forecast, freq="Y")
            monthly_agg = compute_forecast_aggregates(out_forecast, freq="MS")
            annual_forecast_csv = save_aggregated_forecasts(annual_agg, artifacts.run_output_dir, "annual", artifacts.timestamp, logger)
            monthly_forecast_csv = save_aggregated_forecasts(monthly_agg, artifacts.run_output_dir, "monthly", artifacts.timestamp, logger)
            totals_forecast_csv = save_forecast_totals(artifacts.run_output_dir, artifacts.timestamp, out_forecast, logger)
            consistency_forecast_csv = compute_and_save_annual_consistency(
                totals_csv_path=totals_forecast_csv,
                output_dir=artifacts.run_output_dir,
                timestamp=artifacts.timestamp,
                logger=logger,
            )
        except Exception as exc:
            logger.info(f"Failed to compute/save aggregated forecasts: {exc}")
            annual_forecast_csv = None
            monthly_forecast_csv = None
            totals_forecast_csv = None
            consistency_forecast_csv = None

        validation_metrics = extract_best_validation_metrics(model)
        holdout_result = run_holdout_backtest(train_df=train_df, logger=logger, forecast_length=fit_forecast_length)
        holdout_metrics = holdout_result.metrics
        template_artifacts = save_template_artifacts(artifacts.templates_csv, artifacts.templates_json, model, logger)
        save_best_template_artifacts(
            artifacts.best_template_csv,
            artifacts.best_template_json,
            template_artifacts.get("best_template"),
            logger,
        )
        weight_artifacts = build_weight_artifacts(model=model, train_df=train_df)
        logger.info(
            "Prepared weight summaries: "
            f"models={weight_artifacts['model_weight_count']}, "
            f"features={weight_artifacts['feature_weight_count']}, "
            f"parameters={weight_artifacts['parameter_weight_count']}"
        )

        if holdout_result.actual is not None and holdout_result.forecast is not None and holdout_result.lastvalue is not None:
            holdout_horizon = holdout_result.holdout_length
            comparison_title = f"{TARGET_COLUMN} Forecast vs Actual vs LastValueNaive"
            format2_plot = artifacts.run_output_dir / f"AutoTS_forecast_vs_actual_vs_lastvalue_{holdout_horizon}-format2.png"
            legacy_plot = artifacts.run_output_dir / f"forecast_vs_actual_vs_lastvalue_{holdout_horizon}.png"
            pvgis_plot = artifacts.run_output_dir / "PvgisForecast_vs_actual_vs_lastvalue.png"
            pvgis_holdout = build_pvgis_series(
                pvgis_df=pvgis_df,
                target_index=holdout_result.actual.index,
                logger=logger,
            )

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

            if save_pvgis_forecast_vs_actual_vs_lastvalue_plot(
                plot_path=pvgis_plot,
                actual=holdout_result.actual,
                pvgis_forecast=pvgis_holdout,
                lastvalue=holdout_result.lastvalue,
                logger=logger,
            ):
                logger.info(f"Saved PVGIS holdout comparison plot: {pvgis_plot}")

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
            "pvgis_timeseries_file": str(PVGIS_TIMESERIES_CSV),
            "target_column": TARGET_COLUMN,
            "fit_forecast_length": fit_forecast_length,
            "fit_forecast_length_mode": fit_forecast_length_mode,
            "fit_forecast_length_raw": FIT_FORECAST_LENGTH,
            "fit_regressors": ",".join(FIT_REGRESSOR_COLUMNS),
            "train_rows": len(train_df),
            "forecast_length": FORECAST_LENGTH,
            "forecast_start": str(forecast_index.min().date()),
            "forecast_end": str(forecast_index.max().date()),
            "frequency": FREQUENCY,
            "prediction_interval": PREDICTION_INTERVAL,
            "max_generations": MAX_GENERATIONS,
            "max_generations_used": int(MAX_GENERATIONS),
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
            "templates_csv": str(artifacts.templates_csv),
            "templates_json": str(artifacts.templates_json),
            "best_template_csv": str(artifacts.best_template_csv),
            "best_template_json": str(artifacts.best_template_json),
            "template_rows": template_artifacts.get("row_count", 0),
            "weights_csv": str(artifacts.weights_csv),
            "weights_json": str(artifacts.weights_json),
            "model_weight_rows": weight_artifacts.get("model_weight_count", 0),
            "feature_weight_rows": weight_artifacts.get("feature_weight_count", 0),
            "parameter_weight_rows": weight_artifacts.get("parameter_weight_count", 0),
            "annual_forecast_csv": str(annual_forecast_csv) if annual_forecast_csv is not None else "",
            "monthly_forecast_csv": str(monthly_forecast_csv) if monthly_forecast_csv is not None else "",
            "totals_forecast_csv": str(totals_forecast_csv) if totals_forecast_csv is not None else "",
            "consistency_csv": str(consistency_forecast_csv) if consistency_forecast_csv is not None else "",
            "max_generations_used": int(MAX_GENERATIONS),
        }
        pd.DataFrame([metrics_payload]).to_csv(artifacts.metrics_csv, index=False, encoding="utf-8-sig")
        logger.info(f"Saved metrics CSV: {artifacts.metrics_csv}")

        settings_payload = {
            "timestamp": artifacts.timestamp,
            "random_seed": args.random_seed,
            "train_file": str(TRAIN_CSV),
            "train_file_name": TRAIN_CSV.name,
            "train_columns_used": ["date", TARGET_COLUMN] + list(FIT_REGRESSOR_COLUMNS),
            "future_regressor_file": str(TMY_CSV),
            "future_regressor_file_name": TMY_CSV.name,
            "future_regressor_columns_used": list(FIT_REGRESSOR_COLUMNS),
            "pvgis_timeseries_file": str(PVGIS_TIMESERIES_CSV),
            "pvgis_timeseries_file_name": PVGIS_TIMESERIES_CSV.name,
            "pvgis_column": PVGIS_COLUMN,
            "tmy_file": str(TMY_CSV),
            "target_column": TARGET_COLUMN,
            "fit_forecast_length": fit_forecast_length,
            "fit_forecast_length_mode": fit_forecast_length_mode,
            "fit_forecast_length_raw": FIT_FORECAST_LENGTH,
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
            "output_dir": str(artifacts.run_output_dir),
            "forecast_csv": str(artifacts.forecast_csv),
            "metrics_csv": str(artifacts.metrics_csv),
            "settings_csv": str(artifacts.settings_csv),
            "model_pickle": str(artifacts.model_pickle),
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
            "templates_csv": str(artifacts.templates_csv),
            "templates_json": str(artifacts.templates_json),
            "best_template_csv": str(artifacts.best_template_csv),
            "best_template_json": str(artifacts.best_template_json),
            "template_rows": template_artifacts.get("row_count", 0),
            "weights_csv": str(artifacts.weights_csv),
            "weights_json": str(artifacts.weights_json),
            "model_weight_rows": weight_artifacts.get("model_weight_count", 0),
            "feature_weight_rows": weight_artifacts.get("feature_weight_count", 0),
            "parameter_weight_rows": weight_artifacts.get("parameter_weight_count", 0),
            "annual_forecast_csv": str(annual_forecast_csv) if annual_forecast_csv is not None else "",
            "monthly_forecast_csv": str(monthly_forecast_csv) if monthly_forecast_csv is not None else "",
            "totals_forecast_csv": str(totals_forecast_csv) if totals_forecast_csv is not None else "",
            "consistency_csv": str(consistency_forecast_csv) if consistency_forecast_csv is not None else "",
        }
        save_effective_settings_csv(artifacts.settings_csv, settings_payload, logger)
        save_effective_settings_json(artifacts.settings_json, settings_payload, logger)
        save_weight_artifacts(artifacts.weights_csv, artifacts.weights_json, weight_artifacts, logger)

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
    parser = argparse.ArgumentParser(description='10vA1 AutoTS 365-day forecast (child script)')
    parser.add_argument('--random_seed', type=int, default=None, help='Random seed to set for numpy/python random')
    parser.add_argument('--output_dir', default=None, help='Override output directory root for artifacts')
    parser.add_argument('--output_tag', default=None, help='Optional tag to append under output_dir')
    parser.add_argument('--train_csv', default=None, help='Optional override path for training CSV')
    parser.add_argument('--tmy_csv', default=None, help='Optional override path for TMY CSV')
    parser.add_argument('--include_temporal_features', action='store_true', default=False, help='Include temporal features (day_of_year, month, season, etc.)')
    parser.add_argument('--max_generations', type=int, default=MAX_GENERATIONS, help='Override MAX_GENERATIONS for AutoTS')
    parser.add_argument('--fit_forecast_length', default=str(FIT_FORECAST_LENGTH), help="Override FIT_FORECAST_LENGTH (numeric or 'auto')")
    args = parser.parse_args()

    # Allow CLI to override module-level MAX_GENERATIONS (keeps default if not provided)
    try:
        if getattr(args, 'max_generations', None) is not None:
            MAX_GENERATIONS = int(args.max_generations)
    except Exception:
        pass

    # Allow CLI to override FIT_FORECAST_LENGTH (accept numeric or 'auto')
    try:
        if getattr(args, 'fit_forecast_length', None) is not None:
            val = args.fit_forecast_length
            if isinstance(val, str) and val.strip().lower() == 'auto':
                FIT_FORECAST_LENGTH = 'auto'
            else:
                FIT_FORECAST_LENGTH = int(val)
    except Exception:
        pass

    # apply seed if provided
    if args.random_seed is not None:
        try:
            random.seed(int(args.random_seed))
        except Exception:
            pass
        try:
            np.random.seed(int(args.random_seed))
        except Exception:
            pass

    # override paths if provided
    if args.output_dir:
        try:
            OUTPUT_DIR = Path(args.output_dir)
        except Exception:
            pass
    if args.output_tag:
        try:
            OUTPUT_DIR = OUTPUT_DIR / args.output_tag
        except Exception:
            pass
    if args.train_csv:
        try:
            TRAIN_CSV = Path(args.train_csv)
        except Exception:
            pass
    if args.tmy_csv:
        try:
            TMY_CSV = Path(args.tmy_csv)
        except Exception:
            pass

    main(args)
