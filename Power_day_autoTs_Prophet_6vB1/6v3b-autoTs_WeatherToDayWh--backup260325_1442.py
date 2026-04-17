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


def plot_forecast_comparison_legacy(plot_path, index, y_true, y_pred, y_naive,
                                    mase=None, rmsse=None, smape=None,
                                    title=None, figsize=(6, 3), dpi=300):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(index, y_true, label='Actual', color='black', linewidth=2.5)
    plt.plot(index, y_pred, label='AutoTS Forecast', color='dimgray', linewidth=2.5)
    plt.plot(index, y_naive, label='Naive Lag-1', color='gray', linewidth=2, linestyle='--')
    plt.title(title or 'Wh Forecast vs Actual vs Naive Lag-1', fontsize=15, pad=12)
    plt.xlabel('Date', fontsize=13)
    plt.ylabel('Wh', fontsize=13)
    plt.grid(alpha=0.4, linestyle=':', linewidth=0.8)
    plt.xticks(fontsize=11, rotation=30)
    plt.yticks(fontsize=11)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    metrics_parts = []
    if mase is not None:
        metrics_parts.append(f'MASE={mase:.3f}')
    if rmsse is not None:
        metrics_parts.append(f'RMSSE={rmsse:.3f}')
    # if smape is not None:
    #     metrics_parts.append(f'sMAPE={smape:.2f}%')
    metrics_text = '\n'.join(metrics_parts)
    if metrics_text:
        ax.text(1.03, 0.995, metrics_text, transform=ax.transAxes,
                fontsize=10, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 0.60), fontsize=10, frameon=False)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
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
    # 直接固定為 \input\SolarRecord(260310)_d_forWh_WithCodis.csv
    candidate = os.path.normpath(os.path.join(base, 'input', 'SolarRecord(260310)_d_forWh_WithCodis.csv'))
    if os.path.exists(candidate):
        return candidate

    # fallback: 仍保留原來的候選清單
    candidates = [
        os.path.join(base, '..', 'csv', '2000--202602-d-forWh_4b.csv'),
        # os.path.join(base, '..', 'csv', '2000--202602-d-forWh.csv'),
        os.path.join(base, '..', 'csv', 'SolarRecord_260310_1829-daily-1d.csv'),
        os.path.join(base, 'csv', '2000--202602-d-forWh_4b.csv'),
        # os.path.join(base, 'csv', '2000--202602-d-forWh.csv'),
        # os.path.join(base, 'SolarRecord(260204).csv'),
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


def get_output_parent(base: str):
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = pd.Timestamp.now().strftime('%y%m%d_%H%M%S')
    count = 1
    while True:
        output_parent = os.path.join(base, 'output', f'{script_name}_{timestamp}_{count}')
        if not os.path.exists(output_parent):
            break
        count += 1
    os.makedirs(output_parent, exist_ok=True)
    return output_parent, timestamp, count


def init_autots():
    try:
        from autots import AutoTS
    except Exception:
        print('AutoTS 未安裝，請執行: pip install autots')
        raise

    try:
        try:
            import prophet  # type: ignore
        except Exception:
            import fbprophet  # type: ignore
    except Exception:
        print('Prophet 未安裝，請執行: pip install prophet （或 pip install fbprophet 舊名）')
        raise

    return AutoTS


def save_effective_settings(output_dir, inp, horizons, timestamp):
    settings = {
        'input_file': inp,
        'horizons': horizons,
        'ats_kwargs_template': {
            'model_list': ['Prophet'],
            'frequency': 'D',
            'transformer_list': 'default',
            'n_jobs': -1,
            'max_generations': 1,
            'num_validations': 2,
            'validation_method': 'backwards',
            'ensemble': None,
            'prediction_interval': 0.9,
        },
        'output_dir': output_dir,
        'timestamp': timestamp,
    }
    ef_path = os.path.join(output_dir, 'effective_settings.json')
    with open(ef_path, 'w', encoding='utf-8') as ef:
        json.dump(settings, ef, ensure_ascii=False, indent=2)
    print('Saved effective settings to', ef_path)


def run_horizon(AutoTS, ser, horizon, output_dir, timestamp, count, n_jobs, max_generations, transformer_list, model_list, ensemble):
    if len(ser) <= horizon:
        print(f'資料筆數太少無法切出測試集 (len({len(ser)}) <= horizon {horizon})，跳過')
        return

    train_ser = ser.iloc[:-horizon].copy()
    test_ser = ser.iloc[-horizon:].copy()

    dfp_train = train_ser.reset_index()
    dfp_train.columns = ['ds', 'y']
    dfp_train['ds'] = pd.to_datetime(dfp_train['ds'], errors='coerce')
    dfp_train = dfp_train.dropna(subset=['ds'])

    ats_kwargs = dict(
        model_list=model_list,
        forecast_length=horizon,
        frequency='D',
        transformer_list=transformer_list,
        n_jobs=n_jobs,
        max_generations=max_generations,
        num_validations=3,
        validation_method='backwards',
        ensemble=ensemble,
        prediction_interval=0.9,
    )
    print(f'Instantiating AutoTS with model_list=["Prophet"] and horizon={horizon}...')
    model = AutoTS(**{k: v for k, v in ats_kwargs.items() if k != 'forecast_length'})
    try:
        model.forecast_length = horizon
    except Exception:
        pass

    forecast_df = None
    # 先嘗試使用 AutoTS fit + predict；失敗時回退至直接使用 Prophet
    try:
        print(f'Fitting AutoTS (Prophet only) on training set (horizon={horizon})...')
        model = model.fit(dfp_train, date_col='ds', value_col='y')
        print('Predicting', horizon, 'steps ahead...')
        pred = model.predict(forecast_length=horizon)
        if hasattr(pred, 'forecast'):
            forecast_df = pred.forecast
        elif isinstance(pred, pd.DataFrame):
            forecast_df = pred
        else:
            forecast_df = pd.DataFrame(pred)
    except Exception as e:
        print('AutoTS fit/predict failed:', str(e))
        print('嘗試使用 Prophet 直接回退預測...')
        # Prophet 回退
        try:
            try:
                from prophet import Prophet as _Prophet  # type: ignore
            except Exception:
                from fbprophet import Prophet as _Prophet  # type: ignore

            def fallback_prophet_predict(dfp_train_local, horizon_local, ProphetClass):
                dfp = dfp_train_local[['ds', 'y']].dropna()
                m = ProphetClass()
                m.fit(dfp)
                future = m.make_future_dataframe(periods=horizon_local, freq='D')
                fc = m.predict(future)
                fc_tail = fc[['ds', 'yhat']].set_index('ds').tail(horizon_local)
                fc_tail = fc_tail.rename(columns={'yhat': 'Wh'})
                return fc_tail

            forecast_df = fallback_prophet_predict(dfp_train, horizon, _Prophet)
        except Exception as e2:
            print('Prophet fallback 也失敗:', str(e2))
            print('跳過此 horizon', horizon)
            return

    horizon_dir = os.path.join(output_dir, str(horizon))
    os.makedirs(horizon_dir, exist_ok=True)

    try:
        forecast_df = forecast_df.copy()
        if len(forecast_df) == len(test_ser):
            forecast_df.index = test_ser.index
        elif not isinstance(forecast_df.index, pd.DatetimeIndex):
            forecast_df.index = pd.date_range(start=pd.Timestamp.now(), periods=len(forecast_df), freq='D')
    except Exception:
        pass

    filename_base = f'autots_prophet_forecast_{horizon}d'
    out_filename = f'{filename_base}_{timestamp}_{count}.csv'
    out_path = os.path.join(horizon_dir, out_filename)

    try:
        out = forecast_df.reset_index().rename(columns={'index': 'Date'})
        out.to_csv(out_path, index=False)
        print('Saved forecast audit copy to', out_path)
    except Exception as e:
        print('儲存預測失敗:', e)

    out_csv = os.path.join(horizon_dir, f'forecast_Wh_autots_{horizon}d.csv')
    try:
        forecast_df.to_csv(out_csv, index=True)
        print('Saved forecast to', out_csv)
    except Exception as e:
        print('Failed to save forecast_Wh_autots output:', e)

    try:
        y_true = test_ser.iloc[:, 0].astype(float).values
        if 'Wh' in forecast_df.columns:
            y_pred = forecast_df['Wh'].astype(float).values
        else:
            y_pred = forecast_df.iloc[:, 0].astype(float).values

        if len(y_pred) == len(y_true):
            y_naive = np.concatenate(([float(train_ser.iloc[-1, 0])], y_true[:-1])) if horizon > 1 else np.array([float(train_ser.iloc[-1, 0])])
            scores = compute_forecast_scores(y_true, y_pred, train_ser.iloc[:, 0].astype(float).values)
            metrics_path = os.path.join(horizon_dir, f'forecast_Wh_metrics_{horizon}d.json')
            with open(metrics_path, 'w', encoding='utf-8') as mf:
                json.dump(scores, mf, ensure_ascii=False, indent=2)
            print('Saved metrics to', metrics_path)

            try:
                plot_path = os.path.join(horizon_dir, f'forecast_vs_actual_vs_naive_lag1_{horizon}.png')
                plot_forecast_comparison(plot_path, test_ser.index, y_true, y_pred, y_naive,
                                         title=f'Wh Forecast vs Actual vs Naive Lag-1 ({horizon}d)')
                print('Saved comparison chart to', plot_path)
            except Exception as e:
                print('Failed to save comparison chart:', e)

            try:
                plot_path2 = os.path.join(horizon_dir, f'forecast_vs_actual_vs_naive_lag1_{horizon}-format2.png')
                plot_forecast_comparison_legacy(plot_path2, test_ser.index, y_true, y_pred, y_naive,
                                                mase=scores.get('MASE_lag1'),
                                                rmsse=scores.get('RMSSE'),
                                                smape=scores.get('SMAPE(%)'),
                                                title=f'Wh Forecast vs Actual vs Naive Lag-1 ({horizon}d) - format2')
                print('Saved comparison chart (format2) to', plot_path2)
            except Exception as e:
                print('Failed to save comparison chart format2:', e)
        else:
            print('Forecast length does not match test length; skipping metric/plot outputs')
    except Exception as e:
        print('Failed to compute metrics or plots:', e)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='AutoTS Prophet multi-horizon runner')
    parser.add_argument('--horizons', nargs='+', type=int, default=[3, 6, 9],
                        help='List of horizons to run, e.g. --horizons 3 6 9')
    parser.add_argument('--n_jobs', type=int, default=-1, help='AutoTS n_jobs')
    parser.add_argument('--max_generations', type=int, default=1, help='AutoTS max_generations')
    parser.add_argument('--transformer_list', type=str, default='default', help='AutoTS transformer_list')
    parser.add_argument('--model_list', nargs='+', default=['Prophet'], help='AutoTS model_list')
    parser.add_argument('--ensemble', default=None, help='AutoTS ensemble parameter (None, auto, simple, etc.)')
    parser.add_argument('--input_file', type=str, default=None, help='Path to input CSV file')
    parser.add_argument('--loop', action='store_true', help='Run horizons repeatedly until interrupted')
    args = parser.parse_args()

    base = os.path.dirname(__file__)
    if args.input_file:
        inp = os.path.normpath(args.input_file)
        if not os.path.exists(inp):
            print('指定的 input_file 不存在:', inp)
            sys.exit(2)
    else:
        inp = find_input_file(base)
        if inp is None:
            print('未找到預設 CSV，請放置於 csv/ 或修改腳本。')
            sys.exit(2)

    print('Using input:', inp)

    ser = load_series(inp)
    if ser.empty:
        print('讀入時序資料為空，停止。')
        sys.exit(2)

    horizons = args.horizons
    n_jobs = args.n_jobs
    max_generations = args.max_generations
    transformer_list = args.transformer_list
    model_list = args.model_list
    ensemble = None if args.ensemble in (None, 'None', 'none') else args.ensemble

    output_dir, timestamp, count = get_output_parent(base)

    AutoTS = init_autots()

    iteration = 0
    while True:
        iteration += 1
        print(f'Iteration {iteration} start...')
        for horizon in horizons:
            run_horizon(AutoTS, ser, horizon, output_dir, timestamp, count,
                        n_jobs=n_jobs,
                        max_generations=max_generations,
                        transformer_list=transformer_list,
                        model_list=model_list,
                        ensemble=ensemble)

        if not args.loop:
            break

    save_effective_settings(output_dir, inp, horizons, timestamp)


if __name__ == '__main__':
    main()
