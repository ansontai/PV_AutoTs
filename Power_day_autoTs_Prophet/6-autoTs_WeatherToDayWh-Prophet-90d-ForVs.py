#!/usr/bin/env python3
"""AutoTS wrapper — 只使用 Prophet 模型並輸出 90 天預測。

此檔會找專案中的 Wh 時序（或第一個數值欄位），使用 AutoTS 並強制 model_list=['Prophet']。
輸出檔案：`prophet_autots_forecast_90d.csv` 至本檔同目錄。
"""
from __future__ import annotations
import os
import sys
import json
import pandas as pd


def find_input_file(base: str):
    candidates = [
        os.path.join(base, '..', 'csv', '2000--202602-d-forWh_4b.csv'),
        os.path.join(base, '..', 'csv', '2000--202602-d-forWh.csv'),
        os.path.join(base, '..', 'csv', 'SolarRecord_260310_1829-daily-1d.csv'),
        os.path.join(base, 'csv', '2000--202602-d-forWh_4b.csv'),
        os.path.join(base, 'csv', '2000--202602-d-forWh.csv'),
        os.path.join(base, 'SolarRecord(260204).csv'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_series(path: str):
    df = pd.read_csv(path, low_memory=False)
    # attempt to find a date column
    date_cols = [c for c in df.columns if c.lower() in ('localtime', 'date', 'time', 'timestamp')]
    if date_cols:
        dtc = date_cols[0]
        try:
            df[dtc] = pd.to_datetime(df[dtc], errors='coerce')
            df = df.set_index(dtc)
        except Exception:
            pass
    else:
        first = df.columns[0]
        try:
            parsed = pd.to_datetime(df[first], errors='coerce')
            if parsed.notna().sum() > 0:
                df[first] = parsed
                df = df.set_index(first)
        except Exception:
            pass

    # pick the target column 'Wh' if present
    if 'Wh' in df.columns:
        ser = df[['Wh']].copy()
    else:
        numcols = df.select_dtypes(include='number').columns.tolist()
        if not numcols:
            raise SystemExit('找不到可用的數值欄位來預測 (例如 Wh)')
        ser = df[[numcols[0]]].copy()

    ser = ser.apply(pd.to_numeric, errors='coerce')
    ser = ser.dropna(how='all')
    try:
        ser.index = pd.to_datetime(ser.index)
        ser = ser.sort_index()
    except Exception:
        pass
    return ser


def main():
    base = os.path.dirname(__file__)
    inp = find_input_file(base)
    if inp is None:
        print('未找到預設 CSV，請放置於 csv/ 或修改腳本。')
        sys.exit(2)
    print('Using input:', inp)

    ser = load_series(inp)
    if ser.empty:
        print('讀入時序資料為空，停止。')
        sys.exit(2)

    # prepare dataframe for AutoTS / Prophet: ds/y
    dfp = ser.reset_index()
    dfp.columns = ['ds', 'y']
    dfp['ds'] = pd.to_datetime(dfp['ds'], errors='coerce')
    dfp = dfp.dropna(subset=['ds'])

    try:
        from autots import AutoTS
    except Exception:
        print('AutoTS 未安裝，請執行: pip install autots')
        raise

    # Optional sanity-check: ensure Prophet is importable (Autots uses it)
    try:
        try:
            import prophet  # type: ignore
        except Exception:
            import fbprophet  # type: ignore
    except Exception:
        print('Prophet 未安裝，請執行: pip install prophet （或 pip install fbprophet 舊名）')
        raise

    forecast_length = 90

    ats_kwargs = dict(
        model_list=['Prophet'],
        forecast_length=forecast_length,
        frequency='D',
        transformer_list='default',
        # n_jobs=1,
        n_jobs=-1,
        # max_generations=5,
        max_generations=15,
        num_validations=3,
        validation_method='backwards',
        ensemble=None,
        prediction_interval=0.9,
    )

    print('Instantiating AutoTS with model_list=["Prophet"]...')
    model = AutoTS(**{k: v for k, v in ats_kwargs.items() if k != 'forecast_length'})
    # ensure forecast_length set on model if needed by some AutoTS versions
    try:
        model.forecast_length = forecast_length
    except Exception:
        pass

    print('Fitting AutoTS (Prophet only)...')
    model = model.fit(dfp, date_col='ds', value_col='y')

    print('Predicting', forecast_length, 'steps ahead...')
    pred = model.predict(forecast_length=forecast_length)
    # AutoTS predict returns object with .forecast (DataFrame) in many versions
    forecast_df = None
    if hasattr(pred, 'forecast'):
        forecast_df = pred.forecast
    elif isinstance(pred, pd.DataFrame):
        forecast_df = pred
    else:
        # attempt to coerce
        try:
            forecast_df = pd.DataFrame(pred)
        except Exception:
            raise SystemExit('無法擷取 AutoTS predict 結果')

    # save forecast tail (only future periods)
    out_path = os.path.join(base, 'prophet_autots_forecast_90d.csv')
    try:
        # reset index to include Date column if index is datetime
        if isinstance(forecast_df.index, pd.DatetimeIndex):
            out = forecast_df.reset_index().rename(columns={'index': 'Date'})
        else:
            out = forecast_df.reset_index().rename(columns={'index': 'Date'})
        out.to_csv(out_path, index=False)
        print('Saved forecast to', out_path)
    except Exception as e:
        print('儲存預測失敗:', e)


if __name__ == '__main__':
    main()
