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
import re
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

# ENABLE_ONLY_DOC_HANDLE = True
ENABLE_ONLY_DOC_HANDLE = False

# FIT_FORECAST_LENGTH = "auto"
# FIT_FORECAST_LENGTH = 12
FIT_FORECAST_LENGTH = 120
FORECAST_LENGTH = 365
FREQUENCY = "D"
PREDICTION_INTERVAL = 0.9
# MAX_GENERATIONS = 1
# MAX_GENERATIONS = 2
# MAX_GENERATIONS = 15
# MAX_GENERATIONS = 30
MAX_GENERATIONS = 50

NUM_VALIDATIONS = 3
VALIDATION_METHOD = "backwards"
NO_NEGATIVES = True
# ENSEMBLE = "all"
ENSEMBLE = [
    "simple",           # ✅最穩定 baseline
    "distance",         # ✅長短期拆分，長期預測有效
    "simple,distance",  # ✅兩者混合，綜合最佳
]


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
# PVGIS_TIMESERIES_CSV = INPUT_DIR / "Timeseries_24.148_120.703_E5_0kWp_crystSi_25_35deg_1deg_2005_2005[UTC+8][daily][scaled][dateAdj].csv"
PVGIS_TIMESERIES_CSV = INPUT_DIR / "Timeseries_24.148_120.703_E5_0kWp_crystSi_25_35deg_1deg_2005_2005[UTC+8][daily][scaled][vB][dateAdj].csv"
PVGIS_COLUMN = "P_Wh_min_max_scaled"
PVGIS_RAW_P_COLUMN = "P"
PVGIS_E_DAY_COLUMN = "E_day_kWh"

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

# ENSEMBLE_FALLBACK_SAFE = "simple"
# ENSEMBLE_FALLBACK_SAFE = ["simple", "distance"]
# ENSEMBLE_FALLBACK_SAFE = [
#     "simple",           # ✅最穩定 baseline
#     "distance",         # ✅長短期拆分，長期預測有效
#     "simple,distance",  # ✅兩者混合，綜合最佳
#     "subsample",        # ✅對少資料穩定（重要）# XX
#     "simple,subsample"  # ✅穩定增強版
# ]

ENSEMBLE_FALLBACK_SAFE = [
    "simple",           # ✅最穩定 baseline
    "distance",         # ✅長短期拆分，長期預測有效
    "simple,distance",  # ✅兩者混合，綜合最佳
]

@dataclass
class RunArtifacts:
    timestamp: str
    run_output_dir: Path
    forecast_csv: Path
    forecast_csv_with_exogenous: Path
    forecast_csv_without_exogenous: Path
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
    if PVGIS_RAW_P_COLUMN not in df.columns:
        raise ValueError(f"PVGIS timeseries CSV missing required column: {PVGIS_RAW_P_COLUMN}")

    columns = [date_col, PVGIS_COLUMN, PVGIS_RAW_P_COLUMN]
    if PVGIS_E_DAY_COLUMN in df.columns:
        columns.append(PVGIS_E_DAY_COLUMN)

    out = df[columns].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[PVGIS_COLUMN] = pd.to_numeric(out[PVGIS_COLUMN], errors="coerce")
    out[PVGIS_RAW_P_COLUMN] = pd.to_numeric(out[PVGIS_RAW_P_COLUMN], errors="coerce")
    if PVGIS_E_DAY_COLUMN in out.columns:
        out[PVGIS_E_DAY_COLUMN] = pd.to_numeric(out[PVGIS_E_DAY_COLUMN], errors="coerce")

    out = out.dropna(subset=[date_col]).sort_values(date_col)
    out = out.drop_duplicates(subset=[date_col], keep="last")
    out = out.rename(columns={date_col: "date"}).set_index("date")
    out = out.asfreq(FREQUENCY)
    out[PVGIS_COLUMN] = out[PVGIS_COLUMN].interpolate(limit_direction="both")
    out[PVGIS_RAW_P_COLUMN] = out[PVGIS_RAW_P_COLUMN].interpolate(limit_direction="both")

    logger.info(
        "Loaded PVGIS timeseries data: "
        f"rows={len(out)}, date_range={out.index.min().date()} -> {out.index.max().date()}, "
        f"columns={list(out.columns)}"
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
        raise ValueError("PVGIS P_Wh_min_max_scaled still has NaN after fallback fill.")

    out.name = PVGIS_COLUMN
    return out


def build_pvgis_series_raw_p(
    pvgis_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    logger: Logger,
) -> pd.Series:
    """Build raw PVGIS P series by month-day mapping (year is ignored)."""
    logger.info("Building PVGIS raw P series by month-day mapping (year is ignored).")
    pvgis_tmp = pvgis_df[[PVGIS_RAW_P_COLUMN]].copy()
    pvgis_tmp["md"] = pvgis_tmp.index.strftime("%m-%d")
    by_md = pvgis_tmp.groupby("md")[[PVGIS_RAW_P_COLUMN]].mean()

    md_keys = target_index.strftime("%m-%d")
    out = by_md.reindex(md_keys)[PVGIS_RAW_P_COLUMN]
    out.index = target_index

    if out.isna().any():
        logger.info("PVGIS raw P month-day mapping has missing values; applying ffill/bfill fallback.")

    out = out.ffill().bfill()
    if out.isna().any():
        raise ValueError("PVGIS raw P still has NaN after fallback fill.")

    out.name = PVGIS_RAW_P_COLUMN
    return out


def build_e_day_scaled_series(
    pvgis_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    logger: Logger,
) -> pd.Series | None:
    if PVGIS_E_DAY_COLUMN not in pvgis_df.columns:
        logger.info(f"PVGIS timeseries is missing {PVGIS_E_DAY_COLUMN}; skipping E_day_kWh plot.")
        return None

    logger.info("Building E_day_kWh series by month-day mapping (year is ignored).")
    pvgis_tmp = pvgis_df[[PVGIS_E_DAY_COLUMN]].copy()
    pvgis_tmp[PVGIS_E_DAY_COLUMN] = pd.to_numeric(pvgis_tmp[PVGIS_E_DAY_COLUMN], errors="coerce")
    pvgis_tmp["md"] = pvgis_tmp.index.strftime("%m-%d")
    by_md = pvgis_tmp.groupby("md")[[PVGIS_E_DAY_COLUMN]].mean()

    md_keys = target_index.strftime("%m-%d")
    out = by_md.reindex(md_keys)[PVGIS_E_DAY_COLUMN]
    out.index = target_index

    if out.isna().any():
        logger.info(f"{PVGIS_E_DAY_COLUMN} month-day mapping has missing values; applying ffill/bfill fallback.")

    out = out.ffill().bfill()
    if out.isna().any():
        logger.info(f"{PVGIS_E_DAY_COLUMN} still has NaN after fallback; skipping E_day_kWh plot.")
        return None

    scaled = (out * 1000.0).astype(float)
    scaled.name = PVGIS_E_DAY_COLUMN
    return scaled


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

    # Diagnostic logging: show columns kept, sample head, and uniqueness per column
    try:
        logger.info(f"Predict regressor frame shape={pred_fr.shape}, columns={list(pred_fr.columns)}")
        logger.info(f"Predict regressor nunique: {pred_fr.nunique(dropna=False).to_dict()}")
        logger.info(f"Predict regressor sample:\n{pred_fr.head(5).to_string()}")
    except Exception:
        pass

    return pred_fr


def align_pvgis_to_train(train_df: pd.DataFrame, pvgis_df: pd.DataFrame) -> pd.DataFrame:
    """
    對齐 SolarRecord (train_df) 與 PVGIS (pvgis_df) 資料。
    返回一個包含 date, obs_Wh, pvgis_P 的 DataFrame。
    使用月日對齐（忽略年份）。
    """
    # 重命名列以避免衝突
    train_tmp = train_df[[TARGET_COLUMN]].copy()
    train_tmp.columns = ["obs_Wh"]
    train_tmp["md"] = train_tmp.index.strftime("%m-%d")
    
    pvgis_tmp = pvgis_df[[PVGIS_RAW_P_COLUMN]].copy()
    pvgis_tmp.columns = ["pvgis_P"]
    pvgis_tmp["md"] = pvgis_tmp.index.strftime("%m-%d")
    
    # 按月日分別計算平均值（用於對齐）
    train_by_md = train_tmp.groupby("md")["obs_Wh"].mean()
    pvgis_by_md = pvgis_tmp.groupby("md")["pvgis_P"].mean()
    
    # 合併為一個 DataFrame
    merged = pd.DataFrame({
        "obs_Wh": train_by_md,
        "pvgis_P": pvgis_by_md,
    })
    merged = merged.dropna()
    return merged


def compute_doy_p95_scaling_k(train_df: pd.DataFrame, pvgis_df: pd.DataFrame, logger: Logger) -> pd.Series:
    logger.info("Computing GLOBAL P95 scaling factor and returning DOY series...")
    eps = 1e-9
    merged = align_pvgis_to_train(train_df, pvgis_df)
    if merged.empty:
        scalar = 1.0
    else:
        try:
            p95_obs = float(pd.to_numeric(merged["obs_Wh"], errors="coerce").quantile(0.95))
            p95_pvgis = float(pd.to_numeric(merged["pvgis_P"], errors="coerce").quantile(0.95))
            denom = p95_pvgis if abs(p95_pvgis) > eps else eps
            scalar = p95_obs / denom
            if not np.isfinite(scalar):
                scalar = 1.0
        except Exception:
            scalar = 1.0
    scalar = float(scalar)
    k = pd.Series([scalar] * 366, index=range(1, 367), dtype=float)
    k.name = "P95_scaling_k"
    return k


def compute_doy_mean_scaling_k(train_df: pd.DataFrame, pvgis_df: pd.DataFrame, logger: Logger) -> pd.Series:
    logger.info("Computing GLOBAL mean scaling factor and returning DOY series...")
    eps = 1e-9
    merged = align_pvgis_to_train(train_df, pvgis_df)
    if merged.empty:
        scalar = 1.0
    else:
        try:
            mean_obs = float(pd.to_numeric(merged["obs_Wh"], errors="coerce").mean())
            mean_pvgis = float(pd.to_numeric(merged["pvgis_P"], errors="coerce").mean())
            denom = mean_pvgis if abs(mean_pvgis) > eps else eps
            scalar = mean_obs / denom
            if not np.isfinite(scalar):
                scalar = 1.0
        except Exception:
            scalar = 1.0
    scalar = float(scalar)
    k = pd.Series([scalar] * 366, index=range(1, 367), dtype=float)
    k.name = "mean_scaling_k"
    return k


def compute_doy_median_scaling_k(train_df: pd.DataFrame, pvgis_df: pd.DataFrame, logger: Logger) -> pd.Series:
    logger.info("Computing GLOBAL median scaling factor and returning DOY series...")
    eps = 1e-9
    merged = align_pvgis_to_train(train_df, pvgis_df)
    if merged.empty:
        scalar = 1.0
    else:
        try:
            med_obs = float(pd.to_numeric(merged["obs_Wh"], errors="coerce").median())
            med_pvgis = float(pd.to_numeric(merged["pvgis_P"], errors="coerce").median())
            denom = med_pvgis if abs(med_pvgis) > eps else eps
            scalar = med_obs / denom
            if not np.isfinite(scalar):
                scalar = 1.0
        except Exception:
            scalar = 1.0
    scalar = float(scalar)
    k = pd.Series([scalar] * 366, index=range(1, 367), dtype=float)
    k.name = "median_scaling_k"
    return k


def compute_doy_regression_scaling_k(train_df: pd.DataFrame, pvgis_df: pd.DataFrame, logger: Logger) -> tuple[pd.Series, pd.DataFrame]:
    logger.info("Computing GLOBAL linear regression (slope/intercept) and returning DOY series + coeff_df...")
    eps = 1e-9
    merged = align_pvgis_to_train(train_df, pvgis_df)
    if merged.empty:
        slope = 1.0
        intercept = 0.0
        sample_count = 0
    else:
        merged_num = merged.copy()
        merged_num["obs_Wh"] = pd.to_numeric(merged_num["obs_Wh"], errors="coerce")
        merged_num["pvgis_P"] = pd.to_numeric(merged_num["pvgis_P"], errors="coerce")
        merged_num = merged_num.dropna(subset=["obs_Wh", "pvgis_P"])
        sample_count = int(len(merged_num))
        if sample_count < 2:
            slope = 1.0
            intercept = 0.0
        else:
            try:
                X = merged_num["pvgis_P"].values.astype(float)
                y = merged_num["obs_Wh"].values.astype(float)
                coeffs = np.polyfit(X, y, 1)
                slope = float(coeffs[0])
                intercept = float(coeffs[1])
                # slope = float(np.clip(slope, 0.1, 10.0))
                slope = float(np.clip(slope, 0.000001, 100000.0))
                if not np.isfinite(intercept):
                    intercept = 0.0
            except Exception:
                slope = 1.0
                intercept = 0.0

    slope = float(slope)
    intercept = float(intercept)
    k = pd.Series([slope] * 366, index=range(1, 367), dtype=float)
    k.name = "regression_scaling_k"

    coeff_df = pd.DataFrame({
        "doy": list(range(1, 367)),
        "A_slope": [slope] * 366,
        "B_intercept": [intercept] * 366,
        "sample_count": [sample_count] * 366,
    })
    coeff_df["A_slope"] = pd.to_numeric(coeff_df["A_slope"], errors="coerce").fillna(1.0).astype(float)
    coeff_df["B_intercept"] = pd.to_numeric(coeff_df["B_intercept"], errors="coerce").fillna(0.0).astype(float)
    coeff_df["sample_count"] = pd.to_numeric(coeff_df["sample_count"], errors="coerce").fillna(0).astype(int)
    return k, coeff_df


def build_pvgis_p95_scaled_series(
    pvgis_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    scaling_k: pd.Series,
    logger: Logger,
) -> pd.Series:
    """使用 P95 縮放係數構建 PVGIS 序列。"""
    logger.info("Building PVGIS P95-scaled series by DOY mapping...")
    pvgis_tmp = pvgis_df[[PVGIS_RAW_P_COLUMN]].copy()
    pvgis_tmp["doy"] = pvgis_tmp.index.dayofyear
    # Use the 95th percentile (quantile) per DOY as the base value
    by_doy = pvgis_tmp.groupby("doy")[PVGIS_RAW_P_COLUMN].quantile(0.95)

    doy_keys = target_index.dayofyear
    out = by_doy.reindex(doy_keys).values
    out = pd.Series(out, index=target_index)
    
    # 應用縮放係數
    scale_factors = np.array([scaling_k.get(doy, 1.0) for doy in target_index.dayofyear], dtype=float)
    out = out * scale_factors
    
    if out.isna().any():
        out = out.ffill().bfill()
    if out.isna().any():
        logger.info("Warning: P95-scaled series has NaN after ffill/bfill; filling with 1.0")
        out = out.fillna(1.0)
    
    out.name = "P(PVGIS_adj_P95_scaling)"
    return out


def build_pvgis_mean_scaled_series(
    pvgis_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    scaling_k: pd.Series,
    logger: Logger,
) -> pd.Series:
    """使用平均值縮放係數構建 PVGIS 序列。"""
    logger.info("Building PVGIS mean-scaled series by DOY mapping...")
    pvgis_tmp = pvgis_df[[PVGIS_RAW_P_COLUMN]].copy()
    pvgis_tmp["doy"] = pvgis_tmp.index.dayofyear
    by_doy = pvgis_tmp.groupby("doy")[[PVGIS_RAW_P_COLUMN]].mean()
    
    doy_keys = target_index.dayofyear
    out = by_doy.reindex(doy_keys)[PVGIS_RAW_P_COLUMN].values
    out = pd.Series(out, index=target_index)
    
    scale_factors = np.array([scaling_k.get(doy, 1.0) for doy in target_index.dayofyear], dtype=float)
    out = out * scale_factors
    
    if out.isna().any():
        out = out.ffill().bfill()
    if out.isna().any():
        logger.info("Warning: mean-scaled series has NaN after ffill/bfill; filling with 1.0")
        out = out.fillna(1.0)
    
    out.name = "P(PVGIS_adj_mean_scaling)"
    return out


def build_pvgis_median_scaled_series(
    pvgis_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    scaling_k: pd.Series,
    logger: Logger,
) -> pd.Series:
    """使用中位數縮放係數構建 PVGIS 序列。"""
    logger.info("Building PVGIS median-scaled series by DOY mapping...")
    pvgis_tmp = pvgis_df[[PVGIS_RAW_P_COLUMN]].copy()
    pvgis_tmp["doy"] = pvgis_tmp.index.dayofyear
    # Use the median per DOY as the base value
    by_doy = pvgis_tmp.groupby("doy")[PVGIS_RAW_P_COLUMN].median()

    doy_keys = target_index.dayofyear
    out = by_doy.reindex(doy_keys).values
    out = pd.Series(out, index=target_index)
    
    scale_factors = np.array([scaling_k.get(doy, 1.0) for doy in target_index.dayofyear], dtype=float)
    out = out * scale_factors
    
    if out.isna().any():
        out = out.ffill().bfill()
    if out.isna().any():
        logger.info("Warning: median-scaled series has NaN after ffill/bfill; filling with 1.0")
        out = out.fillna(1.0)
    
    out.name = "P(PVGIS_adj_median_scaling)"
    return out


def build_pvgis_regression_scaled_series(
    pvgis_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    scaling_k: pd.Series,
    coeff_df: pd.DataFrame | None = None,
    logger: Logger | None = None,
) -> pd.Series:
    """使用迴歸縮放係數（斜率 A 與截距 B）構建 PVGIS 序列。

    如果提供 `coeff_df`，則會將 B_intercept 套用到最終序列：out = A * base + B。
    """
    if logger:
        logger.info("Building PVGIS regression-scaled series by DOY mapping...")
    pvgis_tmp = pvgis_df[[PVGIS_RAW_P_COLUMN]].copy()
    pvgis_tmp["doy"] = pvgis_tmp.index.dayofyear
    # base value uses DOY mean of raw P
    by_doy = pvgis_tmp.groupby("doy")[PVGIS_RAW_P_COLUMN].mean()

    doy_keys = target_index.dayofyear
    base_vals = by_doy.reindex(doy_keys).values
    out = pd.Series(base_vals, index=target_index)

    a = np.array([scaling_k.get(doy, 1.0) for doy in target_index.dayofyear], dtype=float)
    if coeff_df is not None:
        intercepts = pd.Series(coeff_df["B_intercept"].values, index=coeff_df["doy"].values)
        b = np.array([intercepts.get(doy, 0.0) for doy in target_index.dayofyear], dtype=float)
    else:
        b = np.zeros_like(a)

    out = a * out + b

    if out.isna().any():
        out = out.ffill().bfill()
    if out.isna().any() and logger is not None:
        logger.info("Warning: regression-scaled series has NaN after ffill/bfill; filling with 1.0")
        out = out.fillna(1.0)

    out.name = "P(PVGIS_adj_regression_scaling)"
    return out


def save_regression_coefficients(
    coeff_df: pd.DataFrame,
    output_dir: Path,
    timestamp: str,
    logger: Logger,
) -> Path:
    """
    將迴歸係數 (A 和 B) 保存到 CSV 文件。
    
    Args:
        coeff_df: 包含 doy, A_slope, B_intercept, sample_count 的 DataFrame
        output_dir: 輸出目錄
        timestamp: 時間戳記（用於文件名）
        logger: Logger 實例
    
    Returns:
        保存的 CSV 文件路徑
    """
    csv_path = output_dir / f"regression_coefficients_DOY_{timestamp}.csv"
    try:
        required_cols = ["doy", "A_slope", "B_intercept", "sample_count"]
        missing_cols = [col for col in required_cols if col not in coeff_df.columns]
        if missing_cols:
            raise ValueError(f"Regression coefficient table is missing columns: {missing_cols}")

        coeff_df = coeff_df[required_cols].copy()
        coeff_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved regression coefficients to: {csv_path}")
        logger.info(f"  Total DOYs: {len(coeff_df)}")
        logger.info(f"  A_slope range: [{coeff_df['A_slope'].min():.4f}, {coeff_df['A_slope'].max():.4f}]")
        logger.info(f"  B_intercept range: [{coeff_df['B_intercept'].min():.4f}, {coeff_df['B_intercept'].max():.4f}]")
        logger.info(f"  Mean sample_count per DOY: {coeff_df['sample_count'].mean():.1f}")
        return csv_path
    except Exception as e:
        logger.info(f"Failed to save regression coefficients: {e}")
        raise


def generate_regression_methodology_doc(output_dir: Path, logger: Logger) -> Path:
    """
    生成迴歸係數計算方法說明文檔。
    
    Args:
        output_dir: 輸出目錄
        logger: Logger 實例
    
    Returns:
        生成的 Markdown 文件路徑
    """
    doc_path = output_dir / "REGRESSION_COEFFICIENTS_METHODOLOGY.md"
    try:
        content = """# PVGIS 迴歸縮放係數計算方法說明

## 計算公式

線性迴歸模型：
```
obs_Wh = A × pvgis_P + B
```

其中：
- `obs_Wh`: 實際觀測功率（訓練數據的 Wh 值）
- `pvgis_P`: PVGIS 原始功率值
- `A`: 迴歸斜率（縮放係數）
- `B`: 迴歸截距

## 數據源

- **訓練數據**: 訓練集時期的實際觀測功率（Wh）
- **PVGIS 數據**: 同期 PVGIS 模型輸出的原始功率值 (`P`)

## 分組方式

- **按照日期序數 (Day-of-Year, DOY)** 進行分組
- 對每個 DOY (1-366) 單獨計算迴歸係數
- 這樣可以捕捉季節性變化對縮放關係的影響

## 計算方法

1. **數據準備**:
   - 將訓練數據和 PVGIS 數據按 DOY 進行對齐
   - 移除包含 NaN 值的行

2. **最小二乘法 (OLS)**:
   - 使用 `numpy.polyfit()` 進行 1 次多項式擬合
   - 同時計算斜率 A 和截距 B

3. **參數限制**:
   - **A_slope (斜率)**: 限制在 [0.1, 10.0] 範圍內，防止極端值
   - **B_intercept (截距)**: 允許自由範圍，無限制
   - **樣本不足**: 若某個 DOY 的樣本數 < 2，使用預設值 A=1.0, B=0.0

## 輸出文件

### `regression_coefficients_DOY_[timestamp].csv`

CSV 文件包含以下欄位：

| 欄位 | 說明 |
|-----|------|
| `doy` | Day-of-Year (1-366) |
| `A_slope` | 迴歸斜率（縮放係數），單位無 |
| `B_intercept` | 迴歸截距，單位無 |
| `sample_count` | 該 DOY 用於計算的樣本數量 |

### 使用方式

在預測時，將 PVGIS 原始功率值應用迴歸係數：
```python
P_adj = A × P_original + B
```

## 驗證檢查

- 確保所有 365 或 366 天都有係數
- 檢查 A_slope 是否在合理範圍內
- 查看 sample_count，判斷係數計算的可靠性
- 對於 sample_count = 0 的 DOY，使用了預設值

## 版本記錄

- **生成日期**: 執行時自動記錄
- **訓練數據**: 參見執行日誌中的訓練集信息
"""
        with open(doc_path, "w", encoding="utf-8-sig") as f:
            f.write(content)
        logger.info(f"Generated regression methodology documentation: {doc_path}")
        return doc_path
    except Exception as e:
        logger.info(f"Failed to generate methodology doc: {e}")
        raise


def sanitize_future_regressor(df, logger=None, context: str = ""):
    """Coerce future regressor columns to numeric (pandas-3 safe) and drop all-NaN cols.

    Returns None if resulting frame has no columns or input is None.
    """
    if df is None:
        return None
    try:
        # Make a copy so we don't mutate the caller's frame
        tmp = df.copy()
        if isinstance(tmp, pd.Series):
            tmp = tmp.to_frame()
        # Coerce all columns to numeric (invalid -> NaN)
        tmp = tmp.apply(lambda s: pd.to_numeric(s, errors="coerce"))
        # Drop columns that are entirely NaN
        tmp = tmp.dropna(axis=1, how="all")
        if tmp.shape[1] == 0:
            if logger:
                logger.info(f"sanitize_future_regressor: {context} -> no numeric regressor columns remain after coercion; returning None")
            return None

        # Diagnostic logging: columns kept, types and sample
        if logger:
            try:
                logger.info(f"sanitize_future_regressor: {context} -> kept columns={list(tmp.columns)}, shape={tmp.shape}")
                logger.info(f"sanitize_future_regressor: {context} -> nunique={tmp.nunique(dropna=False).to_dict()}")
                logger.info(f"sanitize_future_regressor: {context} -> head:\n{tmp.head(3).to_string()}")
            except Exception:
                pass

        return tmp
    except Exception as e:
        if logger:
            logger.info(f"sanitize_future_regressor: {context} -> failed to sanitize future regressor: {e}")
        return df


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

    if metrics_lines:
        ax.text(
            1.04,
            0.98,
            "\n".join(metrics_lines),
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
        )

    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 0.60), fontsize=9, frameon=False)
    fig.subplots_adjust(bottom=0.18, right=0.80)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return True


def generate_plots_for_forecast_365d_csvs(
    out_dir: Path,
    actual_series: pd.Series | None,
    lastvalue_series: pd.Series | None,
    train_series: pd.Series | None,
    logger: Logger,
) -> None:
    """Scan for forecast_365d_*.csv and generate per-column single-line plots.
    
    Generates one plot per column (excluding date/time columns).
    Each plot shows only that column's values as a single line.
    Style matches forecast_365d_future_*.png.
    """
    try:
        for csv_path in sorted(out_dir.glob("forecast_365d_*.csv")):
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                logger.info(f"Failed to read CSV {csv_path}: {exc}")
                continue

            # Try common time column names
            time_col = None
            for candidate in ("period_start", "date", "period"):
                if candidate in df.columns:
                    time_col = candidate
                    break
            if time_col is None:
                logger.info(f"No time column found in {csv_path}; skipping")
                continue

            try:
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
                df = df.set_index(time_col)
            except Exception as exc:
                logger.info(f"Failed to set time index for {csv_path}: {exc}")
                continue

            # Generate a plot for each column
            for col in df.columns:
                try:
                    series = pd.to_numeric(df[col], errors="coerce")
                    if series.isna().all():
                        # Skip columns that are entirely NaN
                        continue
                    
                    # Create safe filename from column name
                    safe_col = re.sub(r"[^0-9A-Za-z_.-]", "_", col)
                    plot_file = out_dir / f"{csv_path.stem}_{safe_col}.png"
                    
                    # Generate single-column plot
                    title = f"{col} Forecast"
                    save_single_column_forecast_plot(
                        plot_path=plot_file,
                        forecast_index=df.index,
                        forecast_values=series.to_numpy(dtype=float, copy=False),
                        column_label=col,
                        title=title,
                        logger=logger,
                    )
                    logger.info(f"Saved single-column plot: {plot_file}")
                except Exception as exc:
                    logger.info(f"Failed to save plot for column {col}: {exc}")
    except Exception as exc:
        logger.info(f"generate_plots_for_forecast_365d_csvs failed: {exc}")


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


def save_single_column_forecast_plot(
    plot_path: Path,
    forecast_index: pd.DatetimeIndex,
    forecast_values: np.ndarray,
    column_label: str = "Forecast",
    title: str = "Forecast",
    logger: Logger | None = None,
) -> bool:
    """Save a single-column forecast plot (each CSV column as a separate plot).
    
    Style matches forecast_365d_future_*.png: single line, no confidence bounds.
    Uses column name as both Y-axis label and legend label.
    """
    if not MATPLOTLIB_AVAILABLE:
        if logger:
            logger.info(f"Matplotlib not available; skipping plot: {plot_path}")
        return False
    
    try:
        fig, ax = plt.subplots(figsize=(6, 3), dpi=300)
        ax.plot(forecast_index, forecast_values, label=column_label, color="dimgray", linewidth=2.5)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Date")
        ax.set_ylabel(column_label)
        ax.grid(alpha=0.35, linestyle=":", linewidth=0.8)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        ax.legend(loc="upper left", fontsize=9, frameon=False)
        fig.subplots_adjust(bottom=0.16)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        return True
    except Exception as exc:
        if logger:
            logger.info(f"Failed to save single-column plot {plot_path}: {exc}")
        return False


def save_pvgis_forecast_vs_actual_vs_lastvalue_plot(
    plot_path: Path,
    actual: pd.Series,
    pvgis_forecast: pd.Series,
    lastvalue: pd.Series,
    logger: Logger,
    train_series: pd.Series | None = None,
    forecast_label: str = "PVGIS Forecast",
    title: str | None = None,
) -> bool:
    if not MATPLOTLIB_AVAILABLE:
        logger.info(f"Matplotlib not available; skipping plot: {plot_path}")
        return False

    actual = actual.astype(float)
    pvgis_forecast = pvgis_forecast.astype(float)
    lastvalue = lastvalue.astype(float)

    fig, ax = plt.subplots(figsize=(6, 3), dpi=300, constrained_layout=True)
    ax.plot(actual.index, actual.values, label="Actual", color="black", linewidth=2.5)
    ax.plot(pvgis_forecast.index, pvgis_forecast.values, label=forecast_label, color="dimgray", linewidth=2.5)
    ax.plot(lastvalue.index, lastvalue.values, label="LastValueNaive", color="gray", linewidth=2, linestyle="--")
    ax.set_title(title or f"{forecast_label} vs Actual vs LastValueNaive", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel(TARGET_COLUMN)
    ax.grid(alpha=0.35, linestyle=":", linewidth=0.8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # Compute and display MASE and RMSE
    metrics_lines: list[str] = []
    try:
        eps = 1e-9
        mae = float(np.mean(np.abs(actual - pvgis_forecast)))
        rmse = float(np.sqrt(np.mean((actual - pvgis_forecast) ** 2)))
        
        # Compute MASE if train_series is available
        if train_series is not None and len(train_series) > 1:
            train_series_float = train_series.astype(float)
            naive_denom = float(np.mean(np.abs(np.diff(train_series_float.values))))
            if naive_denom > eps:
                mase = mae / naive_denom
                metrics_lines.append(f"MASE={mase:.3f}")
        
        metrics_lines.append(f"RMSE={rmse:.3f}")
    except Exception as exc:
        logger.info(f"Failed to compute metrics for PVGIS plot: {exc}")

    if metrics_lines:
        ax.text(
            1.04,
            0.98,
            "\n".join(metrics_lines),
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
        )

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
    use_future_regressor: bool = True,
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
    fit_fr = train_part[FIT_REGRESSOR_COLUMNS] if use_future_regressor else None
    pred_fr = test_part[FIT_REGRESSOR_COLUMNS] if use_future_regressor else None

    # sanitize for pandas-3 compatibility (coerce non-numeric -> NaN, drop all-NaN cols)
    fit_fr = sanitize_future_regressor(fit_fr, logger=logger, context="holdout_fit_fr")
    pred_fr = sanitize_future_regressor(pred_fr, logger=logger, context="holdout_pred_fr")

    # Diagnostic logging for regressors used in backtest
    try:
        logger.info(f"holdout fit_fr type={type(fit_fr)}, shape={(None if fit_fr is None else fit_fr.shape)}")
        logger.info(f"holdout pred_fr type={type(pred_fr)}, shape={(None if pred_fr is None else pred_fr.shape)}")
        if fit_fr is not None:
            logger.info(f"holdout fit_fr nunique: {fit_fr.nunique(dropna=False).to_dict()}")
        if pred_fr is not None:
            logger.info(f"holdout pred_fr nunique: {pred_fr.nunique(dropna=False).to_dict()}")
    except Exception:
        pass

    logger.info(f"Running holdout backtest with holdout_length={holdout_len} days (mode={mode}).")

    backtest_model = AutoTS(
        forecast_length=holdout_len,
        frequency=FREQUENCY,
        prediction_interval=PREDICTION_INTERVAL,
        max_generations=max(3, min(6, MAX_GENERATIONS)),
        num_validations=3,
        validation_method=VALIDATION_METHOD,
        ensemble=ENSEMBLE,
        no_negatives=NO_NEGATIVES,
    )

    try:
        backtest_model = backtest_model.fit(train_target, future_regressor=fit_fr)
        # After fit, log best template/feature info if available
        try:
            logger.info(f"Backtest model fitted. results head:\n{str(_extract_results_frame(backtest_model).head(3))}")
        except Exception:
            pass
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

    # 提取预测值（第一列），并重置索引以匹配 test_part 索引
    bt_fcst_values = bt_fcst[TARGET_COLUMN].values if TARGET_COLUMN in bt_fcst.columns else bt_fcst.iloc[:, 0].values
    bt_fcst_aligned = pd.Series(
        bt_fcst_values[:len(test_part)],  # 取前 holdout_len 个值
        index=test_part.index,
        name=TARGET_COLUMN,
    )
    
    metrics = compute_holdout_metrics(
        actual=test_part[TARGET_COLUMN],
        pred=bt_fcst_aligned,
        train_series=train_part[TARGET_COLUMN],
    )
    metrics["holdout_length"] = float(holdout_len)
    return HoldoutBacktestResult(
        metrics=metrics,
        holdout_length=holdout_len,
        actual=test_part[TARGET_COLUMN].copy(),
        forecast=bt_fcst_aligned.copy(),
        lastvalue=pd.Series(
            np.repeat(float(train_part[TARGET_COLUMN].iloc[-1]), len(test_part)),
            index=test_part.index,
            name=TARGET_COLUMN,
        ),
    )


def run_main_model_validation(
    train_df: pd.DataFrame,
    model: AutoTS,
    logger: Logger,
    validation_length: int = 120,
    use_future_regressor: bool = True,
) -> HoldoutBacktestResult:
    """Run validation on the main model's last N days.
    
    Uses the trained model to predict on the last validation_length days of training data.
    """
    n = len(train_df)
    val_len = int(validation_length)
    
    if n <= val_len + 30:
        logger.info("Not enough history for main model validation; skipping.")
        return HoldoutBacktestResult(
            metrics={
                "holdout_mae": float("nan"),
                "holdout_rmse": float("nan"),
                "holdout_mape_pct": float("nan"),
                "holdout_smape_pct": float("nan"),
                "holdout_mase": float("nan"),
                "holdout_length": float("nan"),
            },
            holdout_length=val_len,
        )
    
    # Split: use all data before last val_len for metrics computation
    train_part = train_df.iloc[:-val_len].copy()
    val_part = train_df.iloc[-val_len:].copy()
    
    val_fr = val_part[FIT_REGRESSOR_COLUMNS] if use_future_regressor else None
    # sanitize validation regressors
    val_fr = sanitize_future_regressor(val_fr, logger=logger, context="validation_val_fr")
    
    logger.info(f"Running main model validation on last {val_len} days.")
    try:
        logger.info(f"validation val_fr type={type(val_fr)}, shape={(None if val_fr is None else val_fr.shape)}")
        if val_fr is not None:
            logger.info(f"validation val_fr nunique: {val_fr.nunique(dropna=False).to_dict()}")
    except Exception:
        pass
    
    try:
        val_pred = model.predict(future_regressor=val_fr)
        val_fcst = val_pred.forecast.copy()
    except Exception as exc:
        logger.info(f"Main model validation failed; metrics set to NaN. reason={exc}")
        return HoldoutBacktestResult(
            metrics={
                "holdout_mae": float("nan"),
                "holdout_rmse": float("nan"),
                "holdout_mape_pct": float("nan"),
                "holdout_smape_pct": float("nan"),
                "holdout_mase": float("nan"),
                "holdout_length": float(val_len),
            },
            holdout_length=val_len,
        )
    
    if TARGET_COLUMN not in val_fcst.columns and len(val_fcst.columns) > 0:
        val_fcst = val_fcst.rename(columns={val_fcst.columns[0]: TARGET_COLUMN})
    
    # 提取预测值（第一列），并重置索引以匹配 val_part 索引
    val_fcst_values = val_fcst[TARGET_COLUMN].values if TARGET_COLUMN in val_fcst.columns else val_fcst.iloc[:, 0].values
    val_fcst_aligned = pd.Series(
        val_fcst_values[:len(val_part)],  # 取前 val_len 个值
        index=val_part.index,
        name=TARGET_COLUMN,
    )
    
    metrics = compute_holdout_metrics(
        actual=val_part[TARGET_COLUMN],
        pred=val_fcst_aligned,
        train_series=train_part[TARGET_COLUMN],
    )
    metrics["holdout_length"] = float(val_len)
    return HoldoutBacktestResult(
        metrics=metrics,
        holdout_length=val_len,
        actual=val_part[TARGET_COLUMN].copy(),
        forecast=val_fcst_aligned.copy(),
        lastvalue=pd.Series(
            np.repeat(float(train_part[TARGET_COLUMN].iloc[-1]), len(val_part)),
            index=val_part.index,
            name=TARGET_COLUMN,
        ),
    )


def make_artifact_paths(
    output_root_dir: Path,
    script_path: Path,
    random_seed: int | str | None,
    output_mode_tag: str | None = None,
) -> RunArtifacts:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_folder = str(random_seed if random_seed is not None else 0)
    run_output_dir = output_root_dir / script_path.stem / seed_folder / timestamp
    if output_mode_tag:
        run_output_dir = run_output_dir / output_mode_tag
    return RunArtifacts(
        timestamp=timestamp,
        run_output_dir=run_output_dir,
        forecast_csv=run_output_dir / f"forecast_365d_{timestamp}.csv",
        forecast_csv_with_exogenous=run_output_dir / f"forecast_365d_with_exogenous_{timestamp}.csv",
        forecast_csv_without_exogenous=run_output_dir / f"forecast_365d_without_exogenous_{timestamp}.csv",
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


def find_latest_saved_forecast_csv(search_root: Path) -> Path | None:
    """Find the newest saved forecast_365d CSV, excluding derived summary files."""
    try:
        candidates = []
        for csv_path in Path(search_root).rglob("forecast_365d_*.csv"):
            name = csv_path.name.lower()
            if any(token in name for token in ("_annual_", "_monthly_", "_totals_", "_consistency_")):
                continue
            candidates.append(csv_path)
    except Exception:
        return None

    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_saved_forecast_frame(csv_path: Path) -> pd.DataFrame:
    """Load a saved forecast CSV and normalize its time and numeric columns."""
    df = pd.read_csv(csv_path)
    time_col = _first_existing_column(df.columns, ["date", "period_start", "period"])
    if time_col is None:
        raise ValueError(f"Saved forecast CSV missing a time column: {csv_path}")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).drop_duplicates(subset=[time_col], keep="last")
    df = df.rename(columns={time_col: "date"}).set_index("date")

    required = ["forecast", "lower_bound", "upper_bound"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Saved forecast CSV missing required forecast columns: {missing}")

    numeric_columns = required + [
        PVGIS_RAW_P_COLUMN,
        PVGIS_COLUMN,
        "P(PVGIS_adj_P95_scaling)",
        "P(PVGIS_adj_mean_scaling)",
        "P(PVGIS_adj_median_scaling)",
        "P(PVGIS_adj_regression_scaling)",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


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


def _extract_feature_weight_rows(
    train_df: pd.DataFrame,
    use_future_regressor: bool = True,
) -> tuple[list[dict[str, object]], dict[str, object]]:
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
    if use_future_regressor:
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


def build_weight_artifacts(model: AutoTS, train_df: pd.DataFrame, use_future_regressor: bool = True) -> dict[str, object]:
    model_rows, model_meta = _extract_model_weight_rows(model)
    feature_rows, feature_meta = _extract_feature_weight_rows(train_df, use_future_regressor=use_future_regressor)
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

    Returns a DataFrame with columns: period_start, total_forecast, total_lower, total_upper, total_PVGIS_P
    """
    if forecast_df is None or len(forecast_df) == 0:
        return pd.DataFrame(columns=["period_start", "total_forecast", "total_lower", "total_upper", "total_PVGIS_P"])

    df = forecast_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    if freq == "Y":
        freq = "YE"

    agg = df[["forecast", "lower_bound", "upper_bound", PVGIS_RAW_P_COLUMN]].resample(freq).sum()
    agg = agg.rename(columns={
        "forecast": "total_forecast",
        "lower_bound": "total_lower",
        "upper_bound": "total_upper",
        PVGIS_RAW_P_COLUMN: "total_PVGIS_P"
    })
    agg = agg.reset_index()
    agg = agg.rename(columns={"date": "period_start"})
    return agg[["period_start", "total_forecast", "total_lower", "total_upper", "total_PVGIS_P"]]


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
                "total_PVGIS_P": float(pd.to_numeric(forecast_df[PVGIS_RAW_P_COLUMN], errors="coerce").sum()),
                "total_P_Wh_min_max_scaled": float(pd.to_numeric(forecast_df[PVGIS_COLUMN], errors="coerce").sum()),
                "total_P(PVGIS_adj_P95_scaling)": float(pd.to_numeric(forecast_df["P(PVGIS_adj_P95_scaling)"], errors="coerce").sum()),
                "total_P(PVGIS_adj_mean_scaling)": float(pd.to_numeric(forecast_df["P(PVGIS_adj_mean_scaling)"], errors="coerce").sum()),
                "total_P(PVGIS_adj_median_scaling)": float(pd.to_numeric(forecast_df["P(PVGIS_adj_median_scaling)"], errors="coerce").sum()),
                "total_P(PVGIS_adj_regression_scaling)": float(pd.to_numeric(forecast_df["P(PVGIS_adj_regression_scaling)"], errors="coerce").sum()),
            }
        ]
    )
    file_path = out_dir / f"forecast_365d_totals_{timestamp}.csv"
    totals_df.to_csv(file_path, index=False, encoding="utf-8-sig")
    logger.info(
        f"Saved forecast totals CSV: {file_path} "
        f"(PVGIS_P={totals_df['total_PVGIS_P'].iloc[0]:.2f}, "
        f"P_Wh_min_max_scaled={totals_df['total_P_Wh_min_max_scaled'].iloc[0]:.2f}, "
        f"mean_scaling={totals_df['total_P(PVGIS_adj_mean_scaling)'].iloc[0]:.2f}, "
        f"median_scaling={totals_df['total_P(PVGIS_adj_median_scaling)'].iloc[0]:.2f}, "
        f"regression_scaling={totals_df['total_P(PVGIS_adj_regression_scaling)'].iloc[0]:.2f})"
    )
    return file_path


def compute_and_save_annual_consistency(totals_csv_path: Path, output_dir: Path, timestamp: str, logger: Logger) -> Path:
    """Compute annual consistency metrics (bias and relative error) and save to CSV.
    
    Annual_Bias = Predicted_Annual (total_forecast) - Reference_Annual (total_P_Wh_min_max_scaled)
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
        if np.isnan(predicted_annual):
            logger.info("Predicted annual value is NaN; skipping consistency metrics.")
            return None

        reference_sources = [
            ("total_PVGIS_P", "total_PVGIS_P"),
            ("total_P_Wh_min_max_scaled", "total_P_Wh_min_max_scaled"),
            ("total_P(PVGIS_adj_P95_scaling)", "total_P(PVGIS_adj_P95_scaling)"),
            ("total_P(PVGIS_adj_mean_scaling)", "total_P(PVGIS_adj_mean_scaling)"),
            ("total_P(PVGIS_adj_median_scaling)", "total_P(PVGIS_adj_median_scaling)"),
            ("total_P(PVGIS_adj_regression_scaling)", "total_P(PVGIS_adj_regression_scaling)"),
        ]

        records: list[dict[str, object]] = []
        missing_sources: list[str] = []
        for reference_data_name, totals_column_name in reference_sources:
            reference_annual = float(pd.to_numeric(row.get(totals_column_name), errors="coerce"))
            if np.isnan(reference_annual):
                missing_sources.append(totals_column_name)
                continue

            if reference_annual == 0:
                logger.info(f"Reference annual value is 0 for {reference_data_name}; cannot compute relative error.")
                return None

            annual_bias = predicted_annual - reference_annual
            relative_error_re = abs(predicted_annual - reference_annual) / abs(reference_annual) * 100.0
            percentage_bias_signed = (predicted_annual - reference_annual) / reference_annual * 100.0
            accuracy_pct = 100.0 - relative_error_re

            records.append(
                {
                    "Reference_data_name": reference_data_name,
                    "Reference_Annual_Wh": reference_annual,
                    "Predicted_Annual_Wh": predicted_annual,
                    "Annual_Bias_Wh": annual_bias,
                    "Relative_Error_RE_%": relative_error_re,
                    "Accuracy_%": accuracy_pct,
                    "Percentage_Bias_%": percentage_bias_signed,
                }
            )

        if missing_sources:
            logger.info(f"Missing reference totals in CSV; skipped: {missing_sources}")

        if not records:
            logger.info("No valid reference totals found; skipping consistency metrics.")
            return None

        consistency_df = pd.DataFrame(
            records,
            columns=[
                "Reference_data_name",
                "Reference_Annual_Wh",
                "Predicted_Annual_Wh",
                "Annual_Bias_Wh",
                "Relative_Error_RE_%",
                "Accuracy_%",
                "Percentage_Bias_%",
            ],
        )
        
        file_path = output_dir / f"forecast_365d_consistency_{timestamp}.csv"
        consistency_df.to_csv(file_path, index=False, encoding="utf-8-sig")
        predicted_values_match = consistency_df["Predicted_Annual_Wh"].nunique(dropna=False) == 1
        logger.info(
            f"Saved annual consistency metrics CSV: {file_path} "
            f"(rows={len(consistency_df)}, predicted_constant={predicted_values_match})"
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


def fit_autots_with_decreasing_validations(
    model_params: dict,
    train_target: pd.DataFrame,
    future_regressor: pd.DataFrame | None,
    logger: Logger,
    start_validations: int = 3,
) -> AutoTS:
    """
    拟合 AutoTS 模型，如果失败则递减 num_validations 重试。
    
    Args:
        model_params: AutoTS 初始化参数字典
        train_target: 目标时间序列
        future_regressor: 外生变量
        logger: 日志记录器
        start_validations: 起始 num_validations（通常为 3）
    
    Returns:
        已拟合的 AutoTS 模型
    
    Raises:
        异常：当 num_validations 降到 0 仍失败时抛出
    """
    current_validations = start_validations
    
    while current_validations >= 0:
        try:
            logger.info(
                f"Attempting AutoTS fit with num_validations={current_validations}"
            )
            
            # 创建新的 AutoTS 实例，更新 num_validations
            params = model_params.copy()
            params["num_validations"] = current_validations
            fitted_model = AutoTS(**params)
            fitted_model = fitted_model.fit(train_target, future_regressor=future_regressor)
            
            logger.info(f"Successfully fitted with num_validations={current_validations}")
            return fitted_model
        
        except Exception as exc:
            if current_validations > 0:
                logger.info(
                    f"Fit failed with num_validations={current_validations}; "
                    f"retrying with num_validations={current_validations - 1}. "
                    f"Error: {type(exc).__name__}: {str(exc)[:100]}"
                )
                current_validations -= 1
            else:
                logger.info(
                    f"Fit failed with num_validations=0 (final attempt). "
                    f"Error: {type(exc).__name__}: {str(exc)[:100]}"
                )
                raise
    
    raise RuntimeError("AutoTS fitting failed: all num_validations attempts exhausted")


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
    # TMY 外生變數開關（預設使用 TMY；可透過 --no_tmy_exogenous 關閉）
    USE_TMY = not getattr(args, 'no_tmy_exogenous', False)
    if USE_TMY:
        logger.info("[CONFIG] Using TMY as future exogenous regressors")
    else:
        logger.info("[CONFIG] TMY exogenous regressors DISABLED via CLI (--no_tmy_exogenous)")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = make_artifact_paths(
        OUTPUT_DIR,
        Path(__file__),
        getattr(args, 'random_seed', None),
        getattr(args, 'output_mode_tag', None),
    )
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
        pvgis_df = read_pvgis_timeseries_data(PVGIS_TIMESERIES_CSV, logger)

        train_target = train_df[[TARGET_COLUMN]]
        fit_fr = train_df[FIT_REGRESSOR_COLUMNS] if USE_TMY else None
        # sanitize training regressors for AutoTS (pandas-3 safe)
        fit_fr = sanitize_future_regressor(fit_fr, logger=logger, context="main_fit_fr")

        forecast_start = train_df.index.max() + pd.Timedelta(days=1)
        forecast_index = pd.date_range(forecast_start, periods=FORECAST_LENGTH, freq=FREQUENCY)

        if USE_TMY:
            tmy_df = read_tmy_data(TMY_CSV, logger)
            pred_fr = build_predict_regressor(tmy_df=tmy_df, forecast_index=forecast_index, logger=logger)
            pred_fr = sanitize_future_regressor(pred_fr, logger=logger, context="main_pred_fr")
        else:
            tmy_df = None
            pred_fr = None

        train_fit_fr = fit_fr if USE_TMY else None
        predict_fr = pred_fr if USE_TMY else None

        pvgis_365d = build_pvgis_series(pvgis_df=pvgis_df, target_index=forecast_index, logger=logger)
        pvgis_365d_p = build_pvgis_series_raw_p(pvgis_df=pvgis_df, target_index=forecast_index, logger=logger)

        # 計算 4 個 DOY-based 縮放係數
        k_p95 = compute_doy_p95_scaling_k(train_df=train_df, pvgis_df=pvgis_df, logger=logger)
        k_mean = compute_doy_mean_scaling_k(train_df=train_df, pvgis_df=pvgis_df, logger=logger)
        k_median = compute_doy_median_scaling_k(train_df=train_df, pvgis_df=pvgis_df, logger=logger)
        k_regression, regression_coeff_df = compute_doy_regression_scaling_k(train_df=train_df, pvgis_df=pvgis_df, logger=logger)

        # 保存迴歸係數和方法說明
        try:
            save_regression_coefficients(regression_coeff_df, artifacts.run_output_dir, artifacts.timestamp, logger)
            generate_regression_methodology_doc(artifacts.run_output_dir, logger)
        except Exception as e:
            logger.info(f"Warning: Failed to save regression coefficients/documentation: {e}")

        # 構建 4 個新的 PVGIS 調整列
        pvgis_365d_p95 = build_pvgis_p95_scaled_series(pvgis_df=pvgis_df, target_index=forecast_index, scaling_k=k_p95, logger=logger)
        pvgis_365d_mean = build_pvgis_mean_scaled_series(pvgis_df=pvgis_df, target_index=forecast_index, scaling_k=k_mean, logger=logger)
        pvgis_365d_median = build_pvgis_median_scaled_series(pvgis_df=pvgis_df, target_index=forecast_index, scaling_k=k_median, logger=logger)
        pvgis_365d_regression = build_pvgis_regression_scaled_series(pvgis_df=pvgis_df, target_index=forecast_index, scaling_k=k_regression, coeff_df=regression_coeff_df, logger=logger)

        logger.info(
            f"Forecast horizon: {forecast_index.min().date()} -> {forecast_index.max().date()} "
            f"({FORECAST_LENGTH} days)"
        )

        if ENABLE_ONLY_DOC_HANDLE:
            logger.info("ENABLE_ONLY_DOC_HANDLE=True: skipping AutoTS fit/predict and reusing a saved forecast CSV for document handling.")
            # Use CLI-provided path if available, otherwise search for latest
            if getattr(args, 'saved_forecast_csv', None):
                saved_forecast_csv = Path(args.saved_forecast_csv)
                if not saved_forecast_csv.exists():
                    raise FileNotFoundError(f"Specified --saved-forecast-csv does not exist: {saved_forecast_csv}")
            else:
                saved_forecast_csv = find_latest_saved_forecast_csv(artifacts.run_output_dir.parent)
                if saved_forecast_csv is None:
                    raise FileNotFoundError(
                        f"ENABLE_ONLY_DOC_HANDLE=True but no saved forecast_365d_*.csv was found under {artifacts.run_output_dir.parent}"
                    )

            logger.info(f"Doc-only mode: loading saved forecast CSV: {saved_forecast_csv}")
            out_forecast = load_saved_forecast_frame(saved_forecast_csv)
            forecast_index = pd.DatetimeIndex(out_forecast.index)

            fallback_series_map = {
                PVGIS_RAW_P_COLUMN: build_pvgis_series_raw_p(pvgis_df=pvgis_df, target_index=forecast_index, logger=logger),
                PVGIS_COLUMN: build_pvgis_series(pvgis_df=pvgis_df, target_index=forecast_index, logger=logger),
                "P(PVGIS_adj_P95_scaling)": build_pvgis_p95_scaled_series(pvgis_df=pvgis_df, target_index=forecast_index, scaling_k=k_p95, logger=logger),
                "P(PVGIS_adj_mean_scaling)": build_pvgis_mean_scaled_series(pvgis_df=pvgis_df, target_index=forecast_index, scaling_k=k_mean, logger=logger),
                "P(PVGIS_adj_median_scaling)": build_pvgis_median_scaled_series(pvgis_df=pvgis_df, target_index=forecast_index, scaling_k=k_median, logger=logger),
                "P(PVGIS_adj_regression_scaling)": build_pvgis_regression_scaled_series(pvgis_df=pvgis_df, target_index=forecast_index, scaling_k=k_regression, coeff_df=regression_coeff_df, logger=logger),
            }

            for col, series in fallback_series_map.items():
                if col not in out_forecast.columns:
                    out_forecast[col] = series.reindex(out_forecast.index).values
                else:
                    out_forecast[col] = pd.to_numeric(out_forecast[col], errors="coerce")
                    if out_forecast[col].isna().any():
                        out_forecast[col] = out_forecast[col].fillna(series.reindex(out_forecast.index))

            out_forecast = out_forecast.reset_index()
            out_forecast.to_csv(artifacts.forecast_csv, index=False, encoding="utf-8-sig")
            logger.info(f"Saved doc-only forecast CSV copy: {artifacts.forecast_csv} (columns={list(out_forecast.columns)})")

            annual_agg = compute_forecast_aggregates(out_forecast, freq="Y")
            monthly_agg = compute_forecast_aggregates(out_forecast, freq="MS")
            logger.info(f"Annual aggregates include total_PVGIS_P: {annual_agg[['period_start', 'total_PVGIS_P']].to_string()}")
            logger.info(f"Monthly aggregates computed: {len(monthly_agg)} rows with total_PVGIS_P column")
            annual_forecast_csv = save_aggregated_forecasts(annual_agg, artifacts.run_output_dir, "annual", artifacts.timestamp, logger)
            monthly_forecast_csv = save_aggregated_forecasts(monthly_agg, artifacts.run_output_dir, "monthly", artifacts.timestamp, logger)

            try:
                totals_forecast_csv = save_forecast_totals(artifacts.run_output_dir, artifacts.timestamp, out_forecast, logger)
            except Exception as exc:
                logger.info(f"Failed to save forecast totals CSV: {exc}")
                totals_forecast_csv = None

            try:
                consistency_forecast_csv = compute_and_save_annual_consistency(
                    totals_csv_path=totals_forecast_csv,
                    output_dir=artifacts.run_output_dir,
                    timestamp=artifacts.timestamp,
                    logger=logger,
                )
            except Exception as exc:
                logger.info(f"Failed to save annual consistency CSV: {exc}")
                consistency_forecast_csv = None

            try:
                generate_plots_for_forecast_365d_csvs(
                    out_dir=artifacts.run_output_dir,
                    actual_series=None,
                    lastvalue_series=None,
                    train_series=None,
                    logger=logger,
                )
            except Exception as exc:
                logger.info(f"Failed to generate doc-only forecast plots: {exc}")

            logger.info("Doc-only handling completed; AutoTS fit/predict and model artifact generation were skipped.")
            return

        fit_forecast_length, fit_forecast_length_mode = resolve_fit_forecast_length(
            FIT_FORECAST_LENGTH,
            len(train_target),
        )

        logger.info(
            f"Fitting AutoTS model {'with' if USE_TMY else 'without'} future_regressor on training data. "
            f"fit_forecast_length={fit_forecast_length} (mode={fit_forecast_length_mode}, raw={FIT_FORECAST_LENGTH!r}), "
            f"predict_forecast_length={FORECAST_LENGTH}"
        )

        # 主模型拟合：从 num_validations=3 开始，失败则递减
        main_model_params = {
            "forecast_length": fit_forecast_length,
            "frequency": FREQUENCY,
            "prediction_interval": PREDICTION_INTERVAL,
            "max_generations": MAX_GENERATIONS,
            "validation_method": VALIDATION_METHOD,
            "ensemble": ENSEMBLE,
            "no_negatives": NO_NEGATIVES,
        }
        
        try:
            model = fit_autots_with_decreasing_validations(
                model_params=main_model_params,
                train_target=train_target,
                future_regressor=train_fit_fr,
                logger=logger,
                start_validations=3,
            )
        except ValueError as exc:
            msg = str(exc).lower()
            if "forecast_length is too large" in msg:
                logger.info(
                    "Fit failed with forecast_length/CV constraint after retries; "
                    "trying with single-layer fallback (no validations)."
                )
                fallback_params = main_model_params.copy()
                fallback_params["num_validations"] = 0
                model = AutoTS(**fallback_params)
                model = model.fit(train_target, future_regressor=train_fit_fr)
            else:
                raise

        if USE_TMY:
            logger.info("Predicting 365 days with TMY-based future_regressor.")
        else:
            logger.info("Predicting 365 days WITHOUT TMY future_regressor.")
        # Debug prints to help diagnose sliding-window / mosaic ensemble issues
        try:
            logger.info(f"DEBUG: train_target rows={len(train_target)} fit_fr_shape={getattr(fit_fr, 'shape', None)}")
            logger.info(f"DEBUG: model forecast_length param={getattr(model, 'forecast_length', None)}")
            logger.info(f"DEBUG: requested FORECAST_LENGTH={FORECAST_LENGTH}")
            logger.info(f"DEBUG: model.window={getattr(model, 'window', None)}")
            prediction = model.predict(forecast_length=FORECAST_LENGTH, future_regressor=predict_fr)
        except Exception as exc:
            msg = str(exc).lower()
            if "out of bounds for int8" in msg or "overflow" in msg:
                logger.info(
                    "Predict failed on ensemble='all' long horizon; "
                    "retrying with fallback ensemble='simple' to avoid mosaic ensemble."
                )
                fallback_model_params = {
                    "forecast_length": fit_forecast_length,
                    "frequency": FREQUENCY,
                    "prediction_interval": PREDICTION_INTERVAL,
                    "max_generations": MAX_GENERATIONS,
                    "validation_method": VALIDATION_METHOD,
                    "ensemble": ENSEMBLE_FALLBACK_SAFE,
                    "no_negatives": NO_NEGATIVES,
                }
                
                # Fallback 模型拟合：从 num_validations=3 开始，失败则递减到 0
                fallback_model = fit_autots_with_decreasing_validations(
                    model_params=fallback_model_params,
                    train_target=train_target,
                    future_regressor=train_fit_fr,
                    logger=logger,
                    start_validations=3,
                )
                prediction = fallback_model.predict(
                    forecast_length=FORECAST_LENGTH,
                    future_regressor=predict_fr,
                )
                model = fallback_model
            else:
                raise

        # 第二組預測：不使用外生變數
        logger.info("Predicting 365 days WITHOUT future_regressor (exogenous variables disabled).")
        try:
            prediction_no_exogenous = model.predict(forecast_length=FORECAST_LENGTH, future_regressor=None)
        except Exception as exc:
            msg = str(exc).lower()
            if "out of bounds for int8" in msg or "overflow" in msg:
                logger.info(
                    "Predict (no exogenous) failed on ensemble; "
                    "retrying with fallback ensemble='simple'."
                )
                fallback_model_params_no_ex = {
                    "forecast_length": fit_forecast_length,
                    "frequency": FREQUENCY,
                    "prediction_interval": PREDICTION_INTERVAL,
                    "max_generations": MAX_GENERATIONS,
                    "validation_method": VALIDATION_METHOD,
                    "ensemble": ENSEMBLE_FALLBACK_SAFE,
                    "no_negatives": NO_NEGATIVES,
                }
                fallback_model_no_ex = fit_autots_with_decreasing_validations(
                    model_params=fallback_model_params_no_ex,
                    train_target=train_target,
                    future_regressor=fit_fr,
                    logger=logger,
                    start_validations=3,
                )
                prediction_no_exogenous = fallback_model_no_ex.predict(
                    forecast_length=FORECAST_LENGTH,
                    future_regressor=None,
                )
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

        # 第一組：使用外生變數的預測
        out_forecast_with_exogenous = pd.DataFrame(
            {
                "date": forecast_index,
                "forecast": fcst.reindex(forecast_index)[TARGET_COLUMN].values,
                "lower_bound": lower.reindex(forecast_index)[TARGET_COLUMN].values,
                "upper_bound": upper.reindex(forecast_index)[TARGET_COLUMN].values,
                PVGIS_RAW_P_COLUMN: pvgis_365d_p.reindex(forecast_index).values,
                PVGIS_COLUMN: pvgis_365d.reindex(forecast_index).values,
                "P(PVGIS_adj_P95_scaling)": pvgis_365d_p95.reindex(forecast_index).values,
                "P(PVGIS_adj_mean_scaling)": pvgis_365d_mean.reindex(forecast_index).values,
                "P(PVGIS_adj_median_scaling)": pvgis_365d_median.reindex(forecast_index).values,
                "P(PVGIS_adj_regression_scaling)": pvgis_365d_regression.reindex(forecast_index).values,
            }
        )

        # 第二組：不使用外生變數的預測
        fcst_no_ex = prediction_no_exogenous.forecast.copy()
        lower_no_ex = prediction_no_exogenous.lower_forecast.copy()
        upper_no_ex = prediction_no_exogenous.upper_forecast.copy()

        if TARGET_COLUMN not in fcst_no_ex.columns and len(fcst_no_ex.columns) > 0:
            src_no_ex = fcst_no_ex.columns[0]
            fcst_no_ex = fcst_no_ex.rename(columns={src_no_ex: TARGET_COLUMN})
            lower_no_ex = lower_no_ex.rename(columns={lower_no_ex.columns[0]: TARGET_COLUMN})
            upper_no_ex = upper_no_ex.rename(columns={upper_no_ex.columns[0]: TARGET_COLUMN})

        out_forecast_without_exogenous = pd.DataFrame(
            {
                "date": forecast_index,
                "forecast": fcst_no_ex.reindex(forecast_index)[TARGET_COLUMN].values,
                "lower_bound": lower_no_ex.reindex(forecast_index)[TARGET_COLUMN].values,
                "upper_bound": upper_no_ex.reindex(forecast_index)[TARGET_COLUMN].values,
                PVGIS_RAW_P_COLUMN: pvgis_365d_p.reindex(forecast_index).values,
                PVGIS_COLUMN: pvgis_365d.reindex(forecast_index).values,
                "P(PVGIS_adj_P95_scaling)": pvgis_365d_p95.reindex(forecast_index).values,
                "P(PVGIS_adj_mean_scaling)": pvgis_365d_mean.reindex(forecast_index).values,
                "P(PVGIS_adj_median_scaling)": pvgis_365d_median.reindex(forecast_index).values,
                "P(PVGIS_adj_regression_scaling)": pvgis_365d_regression.reindex(forecast_index).values,
            }
        )

        # 保存兩組 CSV
        out_forecast_with_exogenous.to_csv(artifacts.forecast_csv_with_exogenous, index=False, encoding="utf-8-sig")
        logger.info(f"Saved forecast CSV (with exogenous): {artifacts.forecast_csv_with_exogenous} (columns={list(out_forecast_with_exogenous.columns)})")

        out_forecast_without_exogenous.to_csv(artifacts.forecast_csv_without_exogenous, index=False, encoding="utf-8-sig")
        logger.info(f"Saved forecast CSV (without exogenous): {artifacts.forecast_csv_without_exogenous} (columns={list(out_forecast_without_exogenous.columns)})")

        # 同時保存舊的 forecast_csv 為相容性（預設使用有外生變數的版本）
        out_forecast_with_exogenous.to_csv(artifacts.forecast_csv, index=False, encoding="utf-8-sig")
        logger.info(f"Saved forecast CSV (legacy): {artifacts.forecast_csv}")

        # Aggregate daily forecast into annual and monthly totals and save (兩組預測)
        annual_forecast_csv_with_ex = None
        monthly_forecast_csv_with_ex = None
        annual_forecast_csv_without_ex = None
        monthly_forecast_csv_without_ex = None
        totals_forecast_csv_with_ex = None
        totals_forecast_csv_without_ex = None
        consistency_forecast_csv_with_ex = None
        consistency_forecast_csv_without_ex = None

        # 第一組：有外生變數
        try:
            annual_agg_with_ex = compute_forecast_aggregates(out_forecast_with_exogenous, freq="Y")
            monthly_agg_with_ex = compute_forecast_aggregates(out_forecast_with_exogenous, freq="MS")
            logger.info(f"Annual aggregates (with exogenous) include total_PVGIS_P: {annual_agg_with_ex[['period_start', 'total_PVGIS_P']].to_string()}")
            logger.info(f"Monthly aggregates (with exogenous) computed: {len(monthly_agg_with_ex)} rows with total_PVGIS_P column")
            annual_forecast_csv_with_ex = save_aggregated_forecasts(annual_agg_with_ex, artifacts.run_output_dir, "annual_with_exogenous", artifacts.timestamp, logger)
            monthly_forecast_csv_with_ex = save_aggregated_forecasts(monthly_agg_with_ex, artifacts.run_output_dir, "monthly_with_exogenous", artifacts.timestamp, logger)
        except Exception as exc:
            logger.info(f"Failed to compute/save aggregated forecasts (with exogenous): {exc}")

        # 第二組：無外生變數
        try:
            annual_agg_without_ex = compute_forecast_aggregates(out_forecast_without_exogenous, freq="Y")
            monthly_agg_without_ex = compute_forecast_aggregates(out_forecast_without_exogenous, freq="MS")
            logger.info(f"Annual aggregates (without exogenous) include total_PVGIS_P: {annual_agg_without_ex[['period_start', 'total_PVGIS_P']].to_string()}")
            logger.info(f"Monthly aggregates (without exogenous) computed: {len(monthly_agg_without_ex)} rows with total_PVGIS_P column")
            annual_forecast_csv_without_ex = save_aggregated_forecasts(annual_agg_without_ex, artifacts.run_output_dir, "annual_without_exogenous", artifacts.timestamp, logger)
            monthly_forecast_csv_without_ex = save_aggregated_forecasts(monthly_agg_without_ex, artifacts.run_output_dir, "monthly_without_exogenous", artifacts.timestamp, logger)
        except Exception as exc:
            logger.info(f"Failed to compute/save aggregated forecasts (without exogenous): {exc}")

        # 保存總計 (兩組)
        try:
            totals_forecast_csv_with_ex = save_forecast_totals(artifacts.run_output_dir, artifacts.timestamp + "_with_exogenous", out_forecast_with_exogenous, logger)
        except Exception as exc:
            logger.info(f"Failed to save forecast totals CSV (with exogenous): {exc}")
            totals_forecast_csv_with_ex = None

        try:
            totals_forecast_csv_without_ex = save_forecast_totals(artifacts.run_output_dir, artifacts.timestamp + "_without_exogenous", out_forecast_without_exogenous, logger)
        except Exception as exc:
            logger.info(f"Failed to save forecast totals CSV (without exogenous): {exc}")
            totals_forecast_csv_without_ex = None

        # 年度一致性檢查 (兩組)
        try:
            consistency_forecast_csv_with_ex = compute_and_save_annual_consistency(
                totals_csv_path=totals_forecast_csv_with_ex,
                output_dir=artifacts.run_output_dir,
                timestamp=artifacts.timestamp + "_with_exogenous",
                logger=logger,
            )
        except Exception as exc:
            logger.info(f"Failed to save annual consistency CSV (with exogenous): {exc}")
            consistency_forecast_csv_with_ex = None

        try:
            consistency_forecast_csv_without_ex = compute_and_save_annual_consistency(
                totals_csv_path=totals_forecast_csv_without_ex,
                output_dir=artifacts.run_output_dir,
                timestamp=artifacts.timestamp + "_without_exogenous",
                logger=logger,
            )
        except Exception as exc:
            logger.info(f"Failed to save annual consistency CSV (without exogenous): {exc}")
            consistency_forecast_csv_without_ex = None

        # Extract validation metrics from the fitted model (unchanged)
        validation_metrics = extract_best_validation_metrics(model)

        # Use safe holdout backtest that re-fits on train_part to avoid data leakage
        holdout_result = run_holdout_backtest(
            train_df=train_df,
            logger=logger,
            forecast_length=fit_forecast_length,
            use_future_regressor=USE_TMY,
        )
        holdout_metrics = holdout_result.metrics

        # 同樣的安全回測流程用於不使用外生變數的情況（只改 use_future_regressor）
        logger.info("Running holdout validation for predictions WITHOUT exogenous variables.")
        try:
            holdout_result_no_ex = run_holdout_backtest(
                train_df=train_df,
                logger=logger,
                forecast_length=fit_forecast_length,
                use_future_regressor=False,
            )
            holdout_metrics_no_ex = holdout_result_no_ex.metrics
        except Exception as exc:
            logger.info(f"Holdout backtest without exogenous failed or skipped: {exc}")
            holdout_result_no_ex = None
            holdout_metrics_no_ex = {}
        template_artifacts = save_template_artifacts(artifacts.templates_csv, artifacts.templates_json, model, logger)
        save_best_template_artifacts(
            artifacts.best_template_csv,
            artifacts.best_template_json,
            template_artifacts.get("best_template"),
            logger,
        )
        weight_artifacts = build_weight_artifacts(model=model, train_df=train_df, use_future_regressor=USE_TMY)
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
            pvgis_plot = artifacts.run_output_dir / "PvgisForecast(max_normalization)_vs_actual_vs_lastvalue.png"
            e_day_plot = artifacts.run_output_dir / "PvgisForecast_vs_actual_vs_lastvalue.png"
            pvgis_holdout = build_pvgis_series(
                pvgis_df=pvgis_df,
                target_index=holdout_result.actual.index,
                logger=logger,
            )
            e_day_holdout = build_e_day_scaled_series(
                pvgis_df=pvgis_df,
                target_index=holdout_result.actual.index,
                logger=logger,
            )
            # 構建四個新PVGIS調整縮放的holdout驗證序列
            pvgis_p95_holdout = build_pvgis_p95_scaled_series(
                pvgis_df=pvgis_df,
                target_index=holdout_result.actual.index,
                scaling_k=k_p95,
                logger=logger,
            )
            pvgis_mean_holdout = build_pvgis_mean_scaled_series(
                pvgis_df=pvgis_df,
                target_index=holdout_result.actual.index,
                scaling_k=k_mean,
                logger=logger,
            )
            pvgis_median_holdout = build_pvgis_median_scaled_series(
                pvgis_df=pvgis_df,
                target_index=holdout_result.actual.index,
                scaling_k=k_median,
                logger=logger,
            )
            pvgis_regression_holdout = build_pvgis_regression_scaled_series(
                pvgis_df=pvgis_df,
                target_index=holdout_result.actual.index,
                scaling_k=k_regression,
                coeff_df=regression_coeff_df,
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
                train_series=train_df[TARGET_COLUMN],
                forecast_label="PVGIS Forecast",
                title="PvgisForecast(max_normalization) vs Actual vs LastValueNaive",
            ):
                logger.info(f"Saved PVGIS holdout comparison plot: {pvgis_plot}")

            if e_day_holdout is not None and save_pvgis_forecast_vs_actual_vs_lastvalue_plot(
                plot_path=e_day_plot,
                actual=holdout_result.actual,
                pvgis_forecast=e_day_holdout,
                lastvalue=holdout_result.lastvalue,
                logger=logger,
                train_series=train_df[TARGET_COLUMN],
                forecast_label="E_day_kWh × 1000",
                title="PvgisForecast vs Actual vs LastValueNaive",
            ):
                logger.info(f"Saved E_day_kWh holdout comparison plot: {e_day_plot}")

            # 新增四個PVGIS調整縮放方法的對比圖片
            p95_scaling_plot = artifacts.run_output_dir / "PvgisForecast(P(PVGIS_adj_P95_scaling))_vs_actual_vs_lastvalue.png"
            mean_scaling_plot = artifacts.run_output_dir / "PvgisForecast(P(PVGIS_adj_mean_scaling))_vs_actual_vs_lastvalue.png"
            median_scaling_plot = artifacts.run_output_dir / "PvgisForecast(P(PVGIS_adj_median_scaling))_vs_actual_vs_lastvalue.png"
            regression_scaling_plot = artifacts.run_output_dir / "PvgisForecast(P(PVGIS_adj_regression_scaling))_vs_actual_vs_lastvalue.png"

            if save_pvgis_forecast_vs_actual_vs_lastvalue_plot(
                plot_path=p95_scaling_plot,
                actual=holdout_result.actual,
                pvgis_forecast=pvgis_p95_holdout,
                lastvalue=holdout_result.lastvalue,
                logger=logger,
                train_series=train_df[TARGET_COLUMN],
                forecast_label="PVGIS P95 Scaling",
                title="PvgisForecast(P(PVGIS_adj_P95_scaling)) vs Actual vs LastValueNaive",
            ):
                logger.info(f"Saved P95 scaling holdout comparison plot: {p95_scaling_plot}")

            if save_pvgis_forecast_vs_actual_vs_lastvalue_plot(
                plot_path=mean_scaling_plot,
                actual=holdout_result.actual,
                pvgis_forecast=pvgis_mean_holdout,
                lastvalue=holdout_result.lastvalue,
                logger=logger,
                train_series=train_df[TARGET_COLUMN],
                forecast_label="PVGIS Mean Scaling",
                title="PvgisForecast(P(PVGIS_adj_mean_scaling)) vs Actual vs LastValueNaive",
            ):
                logger.info(f"Saved Mean scaling holdout comparison plot: {mean_scaling_plot}")

            if save_pvgis_forecast_vs_actual_vs_lastvalue_plot(
                plot_path=median_scaling_plot,
                actual=holdout_result.actual,
                pvgis_forecast=pvgis_median_holdout,
                lastvalue=holdout_result.lastvalue,
                logger=logger,
                train_series=train_df[TARGET_COLUMN],
                forecast_label="PVGIS Median Scaling",
                title="PvgisForecast(P(PVGIS_adj_median_scaling)) vs Actual vs LastValueNaive",
            ):
                logger.info(f"Saved Median scaling holdout comparison plot: {median_scaling_plot}")

            if save_pvgis_forecast_vs_actual_vs_lastvalue_plot(
                plot_path=regression_scaling_plot,
                actual=holdout_result.actual,
                pvgis_forecast=pvgis_regression_holdout,
                lastvalue=holdout_result.lastvalue,
                logger=logger,
                train_series=train_df[TARGET_COLUMN],
                forecast_label="PVGIS Regression Scaling",
                title="PvgisForecast(P(PVGIS_adj_regression_scaling)) vs Actual vs LastValueNaive",
            ):
                logger.info(f"Saved Regression scaling holdout comparison plot: {regression_scaling_plot}")

        # 第二組預測的對比圖表（不使用外生變數）
        if holdout_result_no_ex is not None and holdout_result_no_ex.actual is not None:
            holdout_horizon_no_ex = holdout_result_no_ex.holdout_length
            comparison_title_no_ex = f"{TARGET_COLUMN} Forecast vs Actual vs LastValueNaive (Without Exogenous)"
            format2_plot_no_ex = artifacts.run_output_dir / f"AutoTS_forecast_vs_actual_vs_lastvalue_{holdout_horizon_no_ex}_without_exogenous-format2.png"
            legacy_plot_no_ex = artifacts.run_output_dir / f"forecast_vs_actual_vs_lastvalue_{holdout_horizon_no_ex}_without_exogenous.png"
            pvgis_plot_no_ex = artifacts.run_output_dir / "PvgisForecast(max_normalization)_vs_actual_vs_lastvalue_without_exogenous.png"
            e_day_plot_no_ex = artifacts.run_output_dir / "PvgisForecast_vs_actual_vs_lastvalue_without_exogenous.png"
            
            # 為 holdout_result_no_ex 構建 PVGIS 序列
            pvgis_holdout_no_ex = build_pvgis_series(
                pvgis_df=pvgis_df,
                target_index=holdout_result_no_ex.actual.index,
                logger=logger,
            )
            e_day_holdout_no_ex = build_e_day_scaled_series(
                pvgis_df=pvgis_df,
                target_index=holdout_result_no_ex.actual.index,
                logger=logger,
            )
            # 構建四個新PVGIS調整縮放的holdout驗證序列 (without exogenous)
            pvgis_p95_holdout_no_ex = build_pvgis_p95_scaled_series(
                pvgis_df=pvgis_df,
                target_index=holdout_result_no_ex.actual.index,
                scaling_k=k_p95,
                logger=logger,
            )
            pvgis_mean_holdout_no_ex = build_pvgis_mean_scaled_series(
                pvgis_df=pvgis_df,
                target_index=holdout_result_no_ex.actual.index,
                scaling_k=k_mean,
                logger=logger,
            )
            pvgis_median_holdout_no_ex = build_pvgis_median_scaled_series(
                pvgis_df=pvgis_df,
                target_index=holdout_result_no_ex.actual.index,
                scaling_k=k_median,
                logger=logger,
            )
            pvgis_regression_holdout_no_ex = build_pvgis_regression_scaled_series(
                pvgis_df=pvgis_df,
                target_index=holdout_result_no_ex.actual.index,
                scaling_k=k_regression,
                coeff_df=regression_coeff_df,
                logger=logger,
            )

            if save_forecast_vs_actual_vs_lastvalue_plot(
                plot_path=format2_plot_no_ex,
                actual=holdout_result_no_ex.actual,
                forecast=holdout_result_no_ex.forecast,
                lastvalue=holdout_result_no_ex.lastvalue,
                title=comparison_title_no_ex,
                logger=logger,
                metrics=holdout_metrics_no_ex,
            ):
                logger.info(f"Saved holdout comparison plot (without exogenous): {format2_plot_no_ex}")

            if save_forecast_vs_actual_vs_lastvalue_plot(
                plot_path=legacy_plot_no_ex,
                actual=holdout_result_no_ex.actual,
                forecast=holdout_result_no_ex.forecast,
                lastvalue=holdout_result_no_ex.lastvalue,
                title=comparison_title_no_ex,
                logger=logger,
                metrics=holdout_metrics_no_ex,
            ):
                logger.info(f"Saved holdout comparison plot (without exogenous): {legacy_plot_no_ex}")

            if save_pvgis_forecast_vs_actual_vs_lastvalue_plot(
                plot_path=pvgis_plot_no_ex,
                actual=holdout_result_no_ex.actual,
                pvgis_forecast=pvgis_holdout_no_ex,
                lastvalue=holdout_result_no_ex.lastvalue,
                logger=logger,
                train_series=train_df[TARGET_COLUMN],
                forecast_label="PVGIS Forecast",
                title="PvgisForecast(max_normalization) vs Actual vs LastValueNaive (Without Exogenous)",
            ):
                logger.info(f"Saved PVGIS holdout comparison plot (without exogenous): {pvgis_plot_no_ex}")

            if e_day_holdout_no_ex is not None and save_pvgis_forecast_vs_actual_vs_lastvalue_plot(
                plot_path=e_day_plot_no_ex,
                actual=holdout_result_no_ex.actual,
                pvgis_forecast=e_day_holdout_no_ex,
                lastvalue=holdout_result_no_ex.lastvalue,
                logger=logger,
                train_series=train_df[TARGET_COLUMN],
                forecast_label="E_day_kWh × 1000",
                title="PvgisForecast vs Actual vs LastValueNaive (Without Exogenous)",
            ):
                logger.info(f"Saved E_day_kWh holdout comparison plot (without exogenous): {e_day_plot_no_ex}")

            # 新增四個PVGIS調整縮放方法的對比圖片 (without exogenous)
            p95_scaling_plot_no_ex = artifacts.run_output_dir / "PvgisForecast(P(PVGIS_adj_P95_scaling))_vs_actual_vs_lastvalue_without_exogenous.png"
            mean_scaling_plot_no_ex = artifacts.run_output_dir / "PvgisForecast(P(PVGIS_adj_mean_scaling))_vs_actual_vs_lastvalue_without_exogenous.png"
            median_scaling_plot_no_ex = artifacts.run_output_dir / "PvgisForecast(P(PVGIS_adj_median_scaling))_vs_actual_vs_lastvalue_without_exogenous.png"
            regression_scaling_plot_no_ex = artifacts.run_output_dir / "PvgisForecast(P(PVGIS_adj_regression_scaling))_vs_actual_vs_lastvalue_without_exogenous.png"

            if save_pvgis_forecast_vs_actual_vs_lastvalue_plot(
                plot_path=p95_scaling_plot_no_ex,
                actual=holdout_result_no_ex.actual,
                pvgis_forecast=pvgis_p95_holdout_no_ex,
                lastvalue=holdout_result_no_ex.lastvalue,
                logger=logger,
                train_series=train_df[TARGET_COLUMN],
                forecast_label="PVGIS P95 Scaling",
                title="PvgisForecast(P(PVGIS_adj_P95_scaling)) vs Actual vs LastValueNaive (Without Exogenous)",
            ):
                logger.info(f"Saved P95 scaling holdout comparison plot (without exogenous): {p95_scaling_plot_no_ex}")

            if save_pvgis_forecast_vs_actual_vs_lastvalue_plot(
                plot_path=mean_scaling_plot_no_ex,
                actual=holdout_result_no_ex.actual,
                pvgis_forecast=pvgis_mean_holdout_no_ex,
                lastvalue=holdout_result_no_ex.lastvalue,
                logger=logger,
                train_series=train_df[TARGET_COLUMN],
                forecast_label="PVGIS Mean Scaling",
                title="PvgisForecast(P(PVGIS_adj_mean_scaling)) vs Actual vs LastValueNaive (Without Exogenous)",
            ):
                logger.info(f"Saved Mean scaling holdout comparison plot (without exogenous): {mean_scaling_plot_no_ex}")

            if save_pvgis_forecast_vs_actual_vs_lastvalue_plot(
                plot_path=median_scaling_plot_no_ex,
                actual=holdout_result_no_ex.actual,
                pvgis_forecast=pvgis_median_holdout_no_ex,
                lastvalue=holdout_result_no_ex.lastvalue,
                logger=logger,
                train_series=train_df[TARGET_COLUMN],
                forecast_label="PVGIS Median Scaling",
                title="PvgisForecast(P(PVGIS_adj_median_scaling)) vs Actual vs LastValueNaive (Without Exogenous)",
            ):
                logger.info(f"Saved Median scaling holdout comparison plot (without exogenous): {median_scaling_plot_no_ex}")

            if save_pvgis_forecast_vs_actual_vs_lastvalue_plot(
                plot_path=regression_scaling_plot_no_ex,
                actual=holdout_result_no_ex.actual,
                pvgis_forecast=pvgis_regression_holdout_no_ex,
                lastvalue=holdout_result_no_ex.lastvalue,
                logger=logger,
                train_series=train_df[TARGET_COLUMN],
                forecast_label="PVGIS Regression Scaling",
                title="PvgisForecast(P(PVGIS_adj_regression_scaling)) vs Actual vs LastValueNaive (Without Exogenous)",
            ):
                logger.info(f"Saved Regression scaling holdout comparison plot (without exogenous): {regression_scaling_plot_no_ex}")

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
            title=f"{TARGET_COLUMN} 365-Day Future Forecast (With Exogenous)",
            logger=logger,
        ):
            logger.info(f"Saved 365-day future forecast plot (with exogenous): {future_forecast_plot}")

        # 第二組：不使用外生變數的未來預測圖
        future_forecast_plot_no_ex = artifacts.run_output_dir / f"forecast_365d_future_without_exogenous_{artifacts.timestamp}.png"
        future_forecast_values_no_ex = fcst_no_ex.reindex(forecast_index)[TARGET_COLUMN].values
        future_lower_values_no_ex = lower_no_ex.reindex(forecast_index)[TARGET_COLUMN].values
        future_upper_values_no_ex = upper_no_ex.reindex(forecast_index)[TARGET_COLUMN].values
        if save_future_forecast_plot(
            plot_path=future_forecast_plot_no_ex,
            forecast_index=forecast_index,
            forecast_values=future_forecast_values_no_ex,
            lower_values=future_lower_values_no_ex,
            upper_values=future_upper_values_no_ex,
            title=f"{TARGET_COLUMN} 365-Day Future Forecast (Without Exogenous)",
            logger=logger,
        ):
            logger.info(f"Saved 365-day future forecast plot (without exogenous): {future_forecast_plot_no_ex}")

        # After saving forecast CSVs, generate per-column PVGIS-style comparison plots
        try:
            generate_plots_for_forecast_365d_csvs(
                out_dir=artifacts.run_output_dir,
                actual_series=holdout_result.actual if 'holdout_result' in locals() else None,
                lastvalue_series=holdout_result.lastvalue if 'holdout_result' in locals() else None,
                train_series=train_df[TARGET_COLUMN] if 'train_df' in locals() else None,
                logger=logger,
            )
        except Exception as exc:
            logger.info(f"Failed to generate per-column forecast_365d plots: {exc}")

        metrics_payload = {
            "timestamp": artifacts.timestamp,
            "train_file": str(TRAIN_CSV),
            "tmy_file": str(TMY_CSV),
            "pvgis_timeseries_file": str(PVGIS_TIMESERIES_CSV),
            "target_column": TARGET_COLUMN,
            "fit_forecast_length": fit_forecast_length,
            "fit_forecast_length_mode": fit_forecast_length_mode,
            "fit_forecast_length_raw": FIT_FORECAST_LENGTH,
            "fit_regressors": ",".join(FIT_REGRESSOR_COLUMNS) if USE_TMY else "",
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
            "annual_forecast_csv": str(annual_forecast_csv_with_ex) if annual_forecast_csv_with_ex is not None else "",
            "monthly_forecast_csv": str(monthly_forecast_csv_with_ex) if monthly_forecast_csv_with_ex is not None else "",
            "totals_forecast_csv": str(totals_forecast_csv_with_ex) if totals_forecast_csv_with_ex is not None else "",
            "consistency_csv": str(consistency_forecast_csv_with_ex) if consistency_forecast_csv_with_ex is not None else "",
            # Metrics and files for WITH exogenous variant
            "annual_forecast_csv_with_exogenous": str(annual_forecast_csv_with_ex) if annual_forecast_csv_with_ex is not None else "",
            "monthly_forecast_csv_with_exogenous": str(monthly_forecast_csv_with_ex) if monthly_forecast_csv_with_ex is not None else "",
            "totals_forecast_csv_with_exogenous": str(totals_forecast_csv_with_ex) if totals_forecast_csv_with_ex is not None else "",
            "consistency_csv_with_exogenous": str(consistency_forecast_csv_with_ex) if consistency_forecast_csv_with_ex is not None else "",
            "holdout_mae_with_exogenous": holdout_metrics.get("holdout_mae", np.nan),
            "holdout_rmse_with_exogenous": holdout_metrics.get("holdout_rmse", np.nan),
            "holdout_mape_pct_with_exogenous": holdout_metrics.get("holdout_mape_pct", np.nan),
            "holdout_mase_with_exogenous": holdout_metrics.get("holdout_mase", np.nan),
            # Metrics and files for WITHOUT exogenous variant
            "annual_forecast_csv_without_exogenous": str(annual_forecast_csv_without_ex) if annual_forecast_csv_without_ex is not None else "",
            "monthly_forecast_csv_without_exogenous": str(monthly_forecast_csv_without_ex) if monthly_forecast_csv_without_ex is not None else "",
            "totals_forecast_csv_without_exogenous": str(totals_forecast_csv_without_ex) if totals_forecast_csv_without_ex is not None else "",
            "consistency_csv_without_exogenous": str(consistency_forecast_csv_without_ex) if consistency_forecast_csv_without_ex is not None else "",
            "holdout_mae_without_exogenous": holdout_metrics_no_ex.get("holdout_mae", np.nan),
            "holdout_rmse_without_exogenous": holdout_metrics_no_ex.get("holdout_rmse", np.nan),
            "holdout_mape_pct_without_exogenous": holdout_metrics_no_ex.get("holdout_mape_pct", np.nan),
            "holdout_mase_without_exogenous": holdout_metrics_no_ex.get("holdout_mase", np.nan),
            "max_generations_used": int(MAX_GENERATIONS),
        }
        pd.DataFrame([metrics_payload]).to_csv(artifacts.metrics_csv, index=False, encoding="utf-8-sig")
        logger.info(f"Saved metrics CSV: {artifacts.metrics_csv}")

        settings_payload = {
            "timestamp": artifacts.timestamp,
            "random_seed": args.random_seed,
                "use_tmy_exogenous": bool(USE_TMY),
            "train_file": str(TRAIN_CSV),
            "train_file_name": TRAIN_CSV.name,
                "train_columns_used": ["date", TARGET_COLUMN] + (list(FIT_REGRESSOR_COLUMNS) if USE_TMY else []),
                "future_regressor_file": str(TMY_CSV) if USE_TMY else "",
            "future_regressor_file_name": TMY_CSV.name if USE_TMY else "",
                "future_regressor_columns_used": list(FIT_REGRESSOR_COLUMNS) if USE_TMY else [],
            "pvgis_timeseries_file": str(PVGIS_TIMESERIES_CSV),
            "pvgis_timeseries_file_name": PVGIS_TIMESERIES_CSV.name,
            "pvgis_column": PVGIS_COLUMN,
            "tmy_file": str(TMY_CSV) if USE_TMY else "",
            "target_column": TARGET_COLUMN,
            "fit_forecast_length": fit_forecast_length,
            "fit_forecast_length_mode": fit_forecast_length_mode,
            "fit_forecast_length_raw": FIT_FORECAST_LENGTH,
            "fit_regressors": ",".join(FIT_REGRESSOR_COLUMNS) if USE_TMY else "",
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
            "output_mode_tag": str(getattr(args, 'output_mode_tag', '') or ""),
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
            "annual_forecast_csv": str(annual_forecast_csv_with_ex) if annual_forecast_csv_with_ex is not None else "",
            "monthly_forecast_csv": str(monthly_forecast_csv_with_ex) if monthly_forecast_csv_with_ex is not None else "",
            "totals_forecast_csv": str(totals_forecast_csv_with_ex) if totals_forecast_csv_with_ex is not None else "",
            "consistency_csv": str(consistency_forecast_csv_with_ex) if consistency_forecast_csv_with_ex is not None else "",
            # Files for WITH exogenous variant
            "forecast_csv_with_exogenous": str(artifacts.forecast_csv_with_exogenous),
            "forecast_csv_without_exogenous": str(artifacts.forecast_csv_without_exogenous),
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
    parser.add_argument('--output_mode_tag', default=None, help='Optional mode tag to include under timestamped run dir (e.g., with_exogenous|without_exogenous)')
    parser.add_argument('--train_csv', default=None, help='Optional override path for training CSV')
    parser.add_argument('--tmy_csv', default=None, help='Optional override path for TMY CSV')
    parser.add_argument('--include_temporal_features', action='store_true', default=False, help='Include temporal features (day_of_year, month, season, etc.)')
    parser.add_argument('--no_tmy_exogenous', action='store_true', default=False, help='Disable using TMY as future exogenous regressors (default: use TMY)')
    parser.add_argument('--max_generations', type=int, default=MAX_GENERATIONS, help='Override MAX_GENERATIONS for AutoTS')
    parser.add_argument('--fit_forecast_length', default=str(FIT_FORECAST_LENGTH), help="Override FIT_FORECAST_LENGTH (numeric or 'auto')")
    parser.add_argument('--doc-only', action='store_true', default=False, help='Skip AutoTS fit/predict and use saved forecast CSV for doc/chart regeneration')
    parser.add_argument('--saved-forecast-csv', default=None, help='Path to saved forecast CSV for doc-only mode')
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

    # Allow CLI to override ENABLE_ONLY_DOC_HANDLE
    if getattr(args, 'doc_only', False):
        ENABLE_ONLY_DOC_HANDLE = True

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
