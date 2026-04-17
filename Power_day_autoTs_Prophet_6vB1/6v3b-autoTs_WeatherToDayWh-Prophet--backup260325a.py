#!/usr/bin/env python3
"""AutoTS wrapper — 只使用 Prophet 模型並輸出 90 天預測。

此檔會找專案中的 Wh 時序（或第一個數值欄位），使用 AutoTS 並強制 model_list=['Prophet']。
輸出檔案：`prophet_autots_forecast_90d.csv` 至本檔同目錄。
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


def plot_forecast_comparison(plot_path, index, y_true, y_pred, y_naive, title=None, figsize=(12, 6), dpi=150):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.plot(index, y_true, label='Actual', linewidth=2)
    plt.plot(index, y_pred, label='AutoTS Forecast', linewidth=2)
    plt.plot(index, y_naive, label='Naive Lag-1', linewidth=2, linestyle='--')
    plt.title(title or 'Wh Forecast vs Actual vs Naive Lag-1')
    plt.xlabel('Date')
    plt.ylabel('Wh')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=dpi)
    plt.close()


def compute_forecast_scores(y_true, y_pred, train_series):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    train_vals = np.asarray(train_series, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = np.mean(np.abs(np.diff(train_vals))) if train_vals.size > 1 else 0.0
    mase = float(mae / denom) if denom != 0 else np.nan
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    denom_rmsse = float(np.sqrt(np.mean(np.diff(train_vals) ** 2))) if train_vals.size > 1 else 0.0
    rmsse = float(rmse / denom_rmsse) if denom_rmsse != 0 else np.nan
    mean_actual = float(np.mean(y_true)) if y_true.size > 0 else 0.0
    nmae = float(mae / mean_actual) if mean_actual != 0 else np.nan
    nrmse = float(rmse / mean_actual) if mean_actual != 0 else np.nan
    smape = float(np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)) * 100)
    nonzero_mask = np.abs(y_true) > 1e-9
    if nonzero_mask.any():
        mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100)
    else:
        mape = np.nan
    r2 = float(r2_score(y_true, y_pred))
    return {
        'MAE': mae,
        'MASE_lag1': float(mase) if not np.isnan(mase) else None,
        'RMSSE': float(rmsse) if not np.isnan(rmsse) else None,
        'nMAE': float(nmae) if not np.isnan(nmae) else None,
        'nRMSE': float(nrmse) if not np.isnan(nrmse) else None,
        'MAPE(%)': float(mape) if not np.isnan(mape) else None,
        'SMAPE(%)': float(smape) if not np.isnan(smape) else None,
        'R2': r2,
    }


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

    forecast_length = 1

    # prepare dataframe for AutoTS / Prophet: ds/y
    if len(ser) <= forecast_length:
        print(f'資料筆數太少無法切出測試集 (len({len(ser)}) <= forecast_length {forecast_length})')
        sys.exit(2)

    train_ser = ser.iloc[:-forecast_length].copy()
    test_ser = ser.iloc[-forecast_length:].copy()

    dfp_train = train_ser.reset_index()
    dfp_train.columns = ['ds', 'y']
    dfp_train['ds'] = pd.to_datetime(dfp_train['ds'], errors='coerce')
    dfp_train = dfp_train.dropna(subset=['ds'])

    dfp_test = test_ser.reset_index()
    dfp_test.columns = ['ds', 'y']
    dfp_test['ds'] = pd.to_datetime(dfp_test['ds'], errors='coerce')

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

    ats_kwargs = dict(
        model_list=['Prophet'],
        forecast_length=forecast_length,
        frequency='D',
        transformer_list='default',
        # n_jobs=1,
        n_jobs=-1,
        # max_generations=5,
        max_generations=1,
        num_validations=2,
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

    print('Fitting AutoTS (Prophet only) on training set...')
    model = model.fit(dfp_train, date_col='ds', value_col='y')

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

    # create outputs under output/{py_filename}_{timestamp}_{count}
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = pd.Timestamp.now().strftime('%y%m%d_%H%M%S')
    count = 1
    while True:
        output_parent = os.path.join(base, 'output', f'{script_name}_{timestamp}_{count}')
        if not os.path.exists(output_parent):
            break
        count += 1
    os.makedirs(output_parent, exist_ok=True)

    output_dir = output_parent

    filename_base = f'autots_prophet_forecast_{forecast_length}d'
    out_filename = f'{filename_base}_{timestamp}_{count}.csv'
    out_path = os.path.join(output_dir, out_filename)

    # Ensure forecast_df index is a Date-like index (use test index when available)
    try:
        forecast_df = forecast_df.copy()
        if 'test_ser' in locals() and len(forecast_df) == len(test_ser):
            forecast_df.index = test_ser.index
        elif not isinstance(forecast_df.index, pd.DatetimeIndex):
            forecast_df.index = pd.date_range(start=pd.Timestamp.now(), periods=len(forecast_df), freq='D')
    except Exception:
        pass

    # save full forecast + audit flat path
    try:
        out = forecast_df.reset_index().rename(columns={'index': 'Date'})
        out.to_csv(out_path, index=False)
        print('Saved forecast audit copy to', out_path)
    except Exception as e:
        print('儲存預測失敗:', e)

    # Save standard named forecast output for compatibility
    horizon = forecast_length
    out_csv = os.path.join(output_dir, f'forecast_Wh_autots_{horizon}d.csv')
    try:
        forecast_df.to_csv(out_csv, index=True)
        print('Saved forecast to', out_csv)
    except Exception as e:
        print('Failed to save forecast_Wh_autots output:', e)

    # attempt to compute evaluation metrics if we have test data
    try:
        # if training/test split exists (best effort)
        ser_values = ser
        if len(ser_values) > horizon:
            train_ser = ser_values.iloc[:-horizon].copy()
            test_ser = ser_values.iloc[-horizon:].copy()
            y_true = test_ser.iloc[:, 0].astype(float).values
            if 'Wh' in forecast_df.columns:
                y_pred = forecast_df['Wh'].astype(float).values
            else:
                y_pred = forecast_df.iloc[:, 0].astype(float).values
            if len(y_pred) == len(y_true):
                y_naive = np.concatenate(([float(train_ser.iloc[-1, 0])], y_true[:-1])) if horizon > 1 else np.array([float(train_ser.iloc[-1, 0])])
                scores = compute_forecast_scores(y_true, y_pred, train_ser.iloc[:, 0].astype(float).values)
                metrics_path = os.path.join(output_dir, f'forecast_Wh_metrics_{horizon}d.json')
                with open(metrics_path, 'w', encoding='utf-8') as mf:
                    json.dump(scores, mf, ensure_ascii=False, indent=2)
                print('Saved metrics to', metrics_path)

                # plot comparison
                try:
                    plot_path = os.path.join(output_dir, f'forecast_vs_actual_vs_naive_lag1_{horizon}.png')
                    plot_forecast_comparison(plot_path, test_ser.index, y_true, y_pred, y_naive,
                                             title=f'Wh Forecast vs Actual vs Naive Lag-1 ({horizon}d)')
                    print('Saved comparison chart to', plot_path)
                except Exception as e:
                    print('Failed to save comparison chart:', e)

                try:
                    plot_path2 = os.path.join(output_dir, f'forecast_vs_actual_vs_naive_lag1_{horizon}-format2.png')
                    plot_forecast_comparison(plot_path2, test_ser.index, y_true, y_pred, y_naive,
                                             title=f'Wh Forecast vs Actual vs Naive Lag-1 ({horizon}d) - format2')
                    print('Saved comparison chart (format2) to', plot_path2)
                except Exception as e:
                    print('Failed to save comparison chart format2:', e)
            else:
                print('Forecast length does not match test length; skipping metric/plot outputs')
        else:
            print('Insufficient data for train/test split; skipping metric/plot outputs')
    except Exception as e:
        print('Failed to compute metrics or plots:', e)

    # effective settings export
    try:
        effective_settings = {
            'input_file': inp,
            'forecast_length': forecast_length,
            'ats_kwargs': ats_kwargs,
            'output_dir': output_dir,
            'timestamp': timestamp,
        }
        ef_path = os.path.join(output_dir, 'effective_settings.json')
        with open(ef_path, 'w', encoding='utf-8') as ef:
            json.dump(effective_settings, ef, ensure_ascii=False, indent=2)
        print('Saved effective settings to', ef_path)
    except Exception as e:
        print('Warning: failed to write effective_settings.json', e)


if __name__ == '__main__':
    main()
