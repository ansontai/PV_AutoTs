#!/usr/bin/env python3
"""Compute forecast metrics for AutoTS forecast CSVs and write JSON outputs.

Usage example:
  python scripts/compute_forecast_metrics.py --truth ../csv/SolarRecord(260310)_d_forWh_WithCodis.csv --forecast-dir . --horizons 90,60,30
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List

import numpy as np
import pandas as pd


def compute_forecast_scores(y_true, y_pred, train_series):
    import numpy as _np
    from sklearn.metrics import mean_absolute_error as _mae, r2_score as _r2

    y_true = _np.asarray(y_true, dtype=float)
    y_pred = _np.asarray(y_pred, dtype=float)
    train_vals = _np.asarray(train_series, dtype=float)

    mae = float(_mae(y_true, y_pred))
    denom = _np.mean(_np.abs(_np.diff(train_vals))) if train_vals.size > 1 else 0.0
    mase = float(mae / denom) if denom != 0 else _np.nan

    rmse = float(_np.sqrt(_np.mean((y_pred - y_true) ** 2)))
    denom_rmsse = float(_np.sqrt(_np.mean(_np.diff(train_vals) ** 2))) if train_vals.size > 1 else 0.0
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

    return scores, mae, mase, rmsse, smape


def find_latest_forecast(forecast_dir: str, horizon: int) -> str | None:
    pattern = os.path.join(forecast_dir, '**', f'forecast_Wh_autots_{horizon}d.csv')
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def read_truth_series(truth_csv: str) -> pd.Series:
    df = pd.read_csv(truth_csv, parse_dates=['LocalTime', 'Date'], dayfirst=False, low_memory=True)
    if 'Date' in df.columns:
        df['__date__'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.set_index('__date__')
    elif 'LocalTime' in df.columns:
        df = df.set_index(pd.to_datetime(df['LocalTime'], errors='coerce'))
    else:
        raise SystemExit('Truth CSV must contain Date or LocalTime column')
    s = pd.to_numeric(df['Wh'], errors='coerce')
    s = s.sort_index().ffill().bfill()
    return s


def get_pred_series(forecast_csv: str) -> pd.Series:
    df = pd.read_csv(forecast_csv, index_col=0, parse_dates=True)
    # Try common column names
    if 'Wh' in df.columns:
        s = df['Wh']
    else:
        # pick first numeric column
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            s = df[numeric_cols[0]]
        else:
            # last resort: first column
            s = df.iloc[:, 0]
    s = s.sort_index().astype(float)
    return s


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--truth', required=True, help='Path to truth CSV (with Date or LocalTime and Wh)')
    p.add_argument('--forecast-dir', default='.', help='Directory to search for forecast CSVs')
    p.add_argument('--horizons', default='90,60,30', help='Comma-separated horizon list')
    args = p.parse_args(argv)

    truth = read_truth_series(args.truth)
    horizons = [int(x.strip()) for x in args.horizons.split(',') if x.strip()]

    for h in horizons:
        fpath = find_latest_forecast(args.forecast_dir, h)
        if fpath is None:
            print(f'No forecast file found for horizon {h}d')
            continue
        print(f'Using forecast file: {fpath}')
        pred = get_pred_series(fpath)
        # align
        common_idx = pred.index.intersection(truth.index)
        if common_idx.empty:
            print(f'No overlapping dates between forecast and truth for horizon {h}d')
            continue
        y_pred = pred.reindex(common_idx).values
        y_true = truth.reindex(common_idx).values
        # training series inferred as truth before forecast start
        train_series = truth.loc[truth.index < common_idx[0]].values
        if train_series.size == 0:
            # fallback: use truth of entire available history
            train_series = truth.values

        scores, mae, mase, rmsse, smape = compute_forecast_scores(y_true, y_pred, train_series)

        out_json = os.path.join(os.path.dirname(fpath), f'forecast_Wh_metrics_{h}d.json')
        try:
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(scores, f, ensure_ascii=False, indent=2)
            print('Wrote metrics to', out_json)
        except Exception as e:
            print('Failed to write metrics JSON:', e)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
