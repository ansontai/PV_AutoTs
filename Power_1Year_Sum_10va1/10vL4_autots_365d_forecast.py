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


# FIT_FORECAST_LENGTH = "auto"
FIT_FORECAST_LENGTH = 12
# FIT_FORECAST_LENGTH = 120
FORECAST_LENGTH = 365
FREQUENCY = "D"
PREDICTION_INTERVAL = 0.9
MAX_GENERATIONS = 2
# MAX_GENERATIONS = 15
# MAX_GENERATIONS = 30
# MAX_GENERATIONS = 50
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
    """
    計算 DOY 級別的 P95 縮放係數。
    k_P95 = P95(obs_Wh) / P95(pvgis_P) (按 DOY 分組)
    """
    logger.info("Computing DOY-based P95 scaling factors...")
    
    # 對齐數據（月日平均）
    merged = align_pvgis_to_train(train_df, pvgis_df)
    
    # 創建包含 DOY 信息的臨時數據
    train_tmp = train_df[[TARGET_COLUMN]].copy()
    train_tmp.columns = ["obs_Wh"]
    train_tmp["doy"] = train_tmp.index.dayofyear
    
    pvgis_tmp = pvgis_df[[PVGIS_RAW_P_COLUMN]].copy()
    pvgis_tmp.columns = ["pvgis_P"]
    pvgis_tmp["doy"] = pvgis_tmp.index.dayofyear
    
    # 按 DOY 分組
    train_by_doy = train_tmp.groupby("doy")["obs_Wh"].quantile(0.95)  # P95
    pvgis_by_doy = pvgis_tmp.groupby("doy")["pvgis_P"].quantile(0.95)  # P95
    
    # 計算比值，避免除以 0
    k = train_by_doy / pvgis_by_doy.replace(0, 1e-9)
    k = k.reindex(range(1, 367)).fillna(1.0)  # 補充缺失 DOY（如 366）
    k = k.fillna(1.0)
    k.name = "P95_scaling_k"
    return k


def compute_doy_mean_scaling_k(train_df: pd.DataFrame, pvgis_df: pd.DataFrame, logger: Logger) -> pd.Series:
    """
    計算 DOY 級別的平均值縮放係數。
    k_mean = mean(obs_Wh) / mean(pvgis_P) (按 DOY 分組)
    """
    logger.info("Computing DOY-based mean scaling factors...")
    
    train_tmp = train_df[[TARGET_COLUMN]].copy()
    train_tmp.columns = ["obs_Wh"]
    train_tmp["doy"] = train_tmp.index.dayofyear
    
    pvgis_tmp = pvgis_df[[PVGIS_RAW_P_COLUMN]].copy()
    pvgis_tmp.columns = ["pvgis_P"]
    pvgis_tmp["doy"] = pvgis_tmp.index.dayofyear
    
    train_by_doy = train_tmp.groupby("doy")["obs_Wh"].mean()
    pvgis_by_doy = pvgis_tmp.groupby("doy")["pvgis_P"].mean()
    
    k = train_by_doy / pvgis_by_doy.replace(0, 1e-9)
    k = k.reindex(range(1, 367)).fillna(1.0)
    k = k.fillna(1.0)
    k.name = "mean_scaling_k"
    return k


def compute_doy_median_scaling_k(train_df: pd.DataFrame, pvgis_df: pd.DataFrame, logger: Logger) -> pd.Series:
    """
    計算 DOY 級別的中位數縮放係數。
    k_median = median(obs_Wh) / median(pvgis_P) (按 DOY 分組)
    """
    logger.info("Computing DOY-based median scaling factors...")
    
    train_tmp = train_df[[TARGET_COLUMN]].copy()
    train_tmp.columns = ["obs_Wh"]
    train_tmp["doy"] = train_tmp.index.dayofyear
    
    pvgis_tmp = pvgis_df[[PVGIS_RAW_P_COLUMN]].copy()
    pvgis_tmp.columns = ["pvgis_P"]
    pvgis_tmp["doy"] = pvgis_tmp.index.dayofyear
    
    train_by_doy = train_tmp.groupby("doy")["obs_Wh"].median()
    pvgis_by_doy = pvgis_tmp.groupby("doy")["pvgis_P"].median()
    
    k = train_by_doy / pvgis_by_doy.replace(0, 1e-9)
    k = k.reindex(range(1, 367)).fillna(1.0)
    k = k.fillna(1.0)
    k.name = "median_scaling_k"
    return k


def compute_doy_regression_scaling_k(train_df: pd.DataFrame, pvgis_df: pd.DataFrame, logger: Logger) -> pd.Series:
    """
    計算 DOY 級別的線性迴歸縮放係數。
    對每個 DOY，計算迴歸 obs_Wh = a * pvgis_P + b，取斜率 a 作為縮放係數。
    """
    logger.info("Computing DOY-based regression scaling factors...")
    
    train_tmp = train_df[[TARGET_COLUMN]].copy()
    train_tmp.columns = ["obs_Wh"]
    train_tmp["doy"] = train_tmp.index.dayofyear
    
    pvgis_tmp = pvgis_df[[PVGIS_RAW_P_COLUMN]].copy()
    pvgis_tmp.columns = ["pvgis_P"]
    pvgis_tmp["doy"] = pvgis_tmp.index.dayofyear
    
    # 合併訓練和 PVGIS 數據
    combined = train_tmp[["obs_Wh", "doy"]].copy()
    combined["pvgis_P"] = pvgis_tmp["pvgis_P"]
    combined = combined.dropna()
    
    def fit_slope(group):
        """對每個 DOY 的數據計算迴歸斜率。"""
        if len(group) < 2:
            return 1.0  # 若樣本不足，使用預設值 1.0
        try:
            X = group["pvgis_P"].values.reshape(-1, 1)
            y = group["obs_Wh"].values
            # 簡單線性迴歸：計算 (X^T X)^-1 X^T y
            from numpy.linalg import lstsq
            coeffs, _, _, _ = lstsq(X, y, rcond=None)
            slope = float(coeffs[0])
            # 避免極端值
            return np.clip(slope, 0.1, 10.0)
        except Exception:
            return 1.0
    
    k = combined.groupby("doy").apply(fit_slope)
    k = k.reindex(range(1, 367)).fillna(1.0)
    k.name = "regression_scaling_k"
    return k


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
    by_doy = pvgis_tmp.groupby("doy")[[PVGIS_RAW_P_COLUMN]].mean()
    
    doy_keys = target_index.dayofyear
    out = by_doy.reindex(doy_keys)[PVGIS_RAW_P_COLUMN].values
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
    by_doy = pvgis_tmp.groupby("doy")[[PVGIS_RAW_P_COLUMN]].mean()
    
    doy_keys = target_index.dayofyear
    out = by_doy.reindex(doy_keys)[PVGIS_RAW_P_COLUMN].values
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
    logger: Logger,
) -> pd.Series:
    """使用迴歸縮放係數構建 PVGIS 序列。"""
    logger.info("Building PVGIS regression-scaled series by DOY mapping...")
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
        logger.info("Warning: regression-scaled series has NaN after ffill/bfill; filling with 1.0")
        out = out.fillna(1.0)
    
    out.name = "P(PVGIS_adj_regression_scaling)"
    return out


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
    """Scan for forecast_365d_*.csv and generate per-column comparison plots.

    For PVGIS-adjusted columns the PVGIS-style plot is used; for other numeric
    columns the generic forecast-vs-actual plot is used.
    """
    try:
        pvgis_adj_cols = {
            "P(PVGIS_adj_P95_scaling)": "PVGIS_P95",
            "P(PVGIS_adj_mean_scaling)": "PVGIS_mean",
            "P(PVGIS_adj_median_scaling)": "PVGIS_median",
            "P(PVGIS_adj_regression_scaling)": "PVGIS_regression",
        }

        for csv_path in sorted(out_dir.glob("forecast_365d_*.csv")):
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                logger.info(f"Failed to read CSV {csv_path}: {exc}")
                continue

            # try common time column names
            time_col = None
            for candidate in ("period_start", "date", "period"):
                if candidate in df.columns:
                    time_col = candidate
                    break
            if time_col is None:
                logger.info(f"No time column found in {csv_path}; skipping")
                continue

            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.set_index(time_col)

            for col in df.columns:
                try:
                    series = pd.to_numeric(df[col], errors="coerce")
                    if series.isna().all():
                        continue
                except Exception:
                    continue

                safe_col = re.sub(r"[^0-9A-Za-z_.-]", "_", col)
                plot_file = out_dir / f"{csv_path.stem}_{safe_col}.png"

                if col in pvgis_adj_cols:
                    label = pvgis_adj_cols[col]
                    title = f"{label} vs Actual vs LastValueNaive"
                    try:
                        save_pvgis_forecast_vs_actual_vs_lastvalue_plot(
                            plot_path=plot_file,
                            actual=actual_series if actual_series is not None else pd.Series(dtype=float),
                            pvgis_forecast=series,
                            lastvalue=lastvalue_series if lastvalue_series is not None else pd.Series(dtype=float),
                            logger=logger,
                            train_series=train_series,
                            forecast_label=label,
                            title=title,
                        )
                        logger.info(f"Saved PVGIS-adjusted plot: {plot_file}")
                    except Exception as exc:
                        logger.info(f"Failed to save PVGIS plot {plot_file}: {exc}")
                else:
                    title = f"{col} Forecast vs Actual vs LastValueNaive"
                    try:
                        save_forecast_vs_actual_vs_lastvalue_plot(
                            plot_path=plot_file,
                            actual=actual_series if actual_series is not None else pd.Series(dtype=float),
                            forecast=series,
                            lastvalue=lastvalue_series if lastvalue_series is not None else pd.Series(dtype=float),
                            title=title,
                            logger=logger,
                            metrics=None,
                        )
                        logger.info(f"Saved forecast column plot: {plot_file}")
                    except Exception as exc:
                        logger.info(f"Failed to save forecast plot {plot_file}: {exc}")
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

    # sanitize for pandas-3 compatibility (coerce non-numeric -> NaN, drop all-NaN cols)
    fit_fr = sanitize_future_regressor(fit_fr, logger=logger, context="holdout_fit_fr")
    pred_fr = sanitize_future_regressor(pred_fr, logger=logger, context="holdout_pred_fr")

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
    
    val_fr = val_part[FIT_REGRESSOR_COLUMNS]
    # sanitize validation regressors
    val_fr = sanitize_future_regressor(val_fr, logger=logger, context="validation_val_fr")
    
    logger.info(f"Running main model validation on last {val_len} days.")
    
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

    Returns a DataFrame with columns: period_start, total_forecast, total_lower, total_upper, total_PVGIS_P
    """
    if forecast_df is None or len(forecast_df) == 0:
        return pd.DataFrame(columns=["period_start", "total_forecast", "total_lower", "total_upper", "total_PVGIS_P"])

    df = forecast_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

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
            }
        ]
    )
    file_path = out_dir / f"forecast_365d_totals_{timestamp}.csv"
    totals_df.to_csv(file_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved forecast totals CSV: {file_path} (PVGIS_P={totals_df['total_PVGIS_P'].iloc[0]:.2f}, P_Wh_min_max_scaled={totals_df['total_P_Wh_min_max_scaled'].iloc[0]:.2f})")
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
        reference_annual = float(pd.to_numeric(row.get("total_P_Wh_min_max_scaled"), errors="coerce"))
        # also capture the raw PVGIS total if present for auditing
        try:
            reference_raw = float(pd.to_numeric(row.get("total_PVGIS_P"), errors="coerce"))
        except Exception:
            reference_raw = float("nan")
        
        if np.isnan(predicted_annual) or np.isnan(reference_annual):
            logger.info("Predicted or reference annual values are NaN; skipping consistency metrics.")
            return None
        
        if reference_annual == 0:
            logger.info("Reference annual value is 0; cannot compute relative error.")
            return None
        
        annual_bias = predicted_annual - reference_annual
        relative_error_re = abs(predicted_annual - reference_annual) / abs(reference_annual) * 100.0
        # Signed percentage bias: (Predicted - Reference) / Reference * 100%
        percentage_bias_signed = (predicted_annual - reference_annual) / reference_annual * 100.0
        # Accuracy as a simple complement: 100% - relative error
        accuracy_pct = 100.0 - relative_error_re

        consistency_df = pd.DataFrame(
            [
                {
                    "Predicted_Annual_Wh": predicted_annual,
                    # keep legacy column name for backwards compatibility
                    "Reference_Annual_Wh": reference_annual,
                    "Reference_Annual_Wh_Adjusted_PVGIS": reference_annual,
                    "Reference_Annual_Wh_Raw_PVGIS": reference_raw,
                    "Annual_Bias_Wh": annual_bias,
                    "Relative_Error_RE_%": relative_error_re,
                    "Accuracy_%": accuracy_pct,
                    "Percentage_Bias_%": percentage_bias_signed,
                }
            ]
        )
        
        file_path = output_dir / f"forecast_365d_consistency_{timestamp}.csv"
        consistency_df.to_csv(file_path, index=False, encoding="utf-8-sig")
        logger.info(
            f"Saved annual consistency metrics CSV: {file_path} "
            f"(Bias={annual_bias:.2f} Wh, RelError={relative_error_re:.2f}%, Accuracy={accuracy_pct:.2f}%, PercBias={percentage_bias_signed:.2f}%)"
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
        pvgis_df = read_pvgis_timeseries_data(PVGIS_TIMESERIES_CSV, logger)

        train_target = train_df[[TARGET_COLUMN]]
        fit_fr = train_df[FIT_REGRESSOR_COLUMNS]
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

        pvgis_365d = build_pvgis_series(pvgis_df=pvgis_df, target_index=forecast_index, logger=logger)
        pvgis_365d_p = build_pvgis_series_raw_p(pvgis_df=pvgis_df, target_index=forecast_index, logger=logger)

        # 計算 4 個 DOY-based 縮放係數
        k_p95 = compute_doy_p95_scaling_k(train_df=train_df, pvgis_df=pvgis_df, logger=logger)
        k_mean = compute_doy_mean_scaling_k(train_df=train_df, pvgis_df=pvgis_df, logger=logger)
        k_median = compute_doy_median_scaling_k(train_df=train_df, pvgis_df=pvgis_df, logger=logger)
        k_regression = compute_doy_regression_scaling_k(train_df=train_df, pvgis_df=pvgis_df, logger=logger)

        # 構建 4 個新的 PVGIS 調整列
        pvgis_365d_p95 = build_pvgis_p95_scaled_series(pvgis_df=pvgis_df, target_index=forecast_index, scaling_k=k_p95, logger=logger)
        pvgis_365d_mean = build_pvgis_mean_scaled_series(pvgis_df=pvgis_df, target_index=forecast_index, scaling_k=k_mean, logger=logger)
        pvgis_365d_median = build_pvgis_median_scaled_series(pvgis_df=pvgis_df, target_index=forecast_index, scaling_k=k_median, logger=logger)
        pvgis_365d_regression = build_pvgis_regression_scaled_series(pvgis_df=pvgis_df, target_index=forecast_index, scaling_k=k_regression, logger=logger)

        logger.info(
            f"Forecast horizon: {forecast_index.min().date()} -> {forecast_index.max().date()} "
            f"({FORECAST_LENGTH} days)"
        )

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
                future_regressor=fit_fr,
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
                model = model.fit(train_target, future_regressor=fit_fr)
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
            prediction = model.predict(forecast_length=FORECAST_LENGTH, future_regressor=pred_fr)
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
                    future_regressor=fit_fr,
                    logger=logger,
                    start_validations=3,
                )
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
                PVGIS_RAW_P_COLUMN: pvgis_365d_p.reindex(forecast_index).values,
                PVGIS_COLUMN: pvgis_365d.reindex(forecast_index).values,
                "P(PVGIS_adj_P95_scaling)": pvgis_365d_p95.reindex(forecast_index).values,
                "P(PVGIS_adj_mean_scaling)": pvgis_365d_mean.reindex(forecast_index).values,
                "P(PVGIS_adj_median_scaling)": pvgis_365d_median.reindex(forecast_index).values,
                "P(PVGIS_adj_regression_scaling)": pvgis_365d_regression.reindex(forecast_index).values,
            }
        )

        out_forecast.to_csv(artifacts.forecast_csv, index=False, encoding="utf-8-sig")
        logger.info(f"Saved forecast CSV: {artifacts.forecast_csv} (columns={list(out_forecast.columns)})")
        logger.info(f"Forecast output includes PVGIS_P raw values (total={out_forecast[PVGIS_RAW_P_COLUMN].sum():.2f})")

        # Aggregate daily forecast into annual and monthly totals and save
        try:
            annual_agg = compute_forecast_aggregates(out_forecast, freq="Y")
            monthly_agg = compute_forecast_aggregates(out_forecast, freq="MS")
            logger.info(f"Annual aggregates include total_PVGIS_P: {annual_agg[['period_start', 'total_PVGIS_P']].to_string()}")
            logger.info(f"Monthly aggregates computed: {len(monthly_agg)} rows with total_PVGIS_P column")
            annual_forecast_csv = save_aggregated_forecasts(annual_agg, artifacts.run_output_dir, "annual", artifacts.timestamp, logger)
            monthly_forecast_csv = save_aggregated_forecasts(monthly_agg, artifacts.run_output_dir, "monthly", artifacts.timestamp, logger)
        except Exception as exc:
            logger.info(f"Failed to compute/save aggregated forecasts: {exc}")
            annual_forecast_csv = None
            monthly_forecast_csv = None

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

        validation_metrics = extract_best_validation_metrics(model)
        holdout_result = run_main_model_validation(train_df=train_df, model=model, logger=logger, validation_length=fit_forecast_length)
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
            pvgis_plot = artifacts.run_output_dir / "PvgisForecast(min_max_scale)_vs_actual_vs_lastvalue.png"
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
                title="PvgisForecast(min_max_scale) vs Actual vs LastValueNaive",
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
                "use_tmy_exogenous": bool(USE_TMY),
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
    parser.add_argument('--no_tmy_exogenous', action='store_true', default=False, help='Disable using TMY as future exogenous regressors (default: use TMY)')
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
