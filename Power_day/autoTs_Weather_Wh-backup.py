import os
import pandas as pd
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    base = os.path.dirname(__file__)
    csv_path = os.path.normpath(os.path.join(base, '..', 'csv', 'SolarRecord(260204)_d_Wh_WithCodis.csv'))
    out_dir = os.path.normpath(os.path.join(base, '..', 'output'))
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path, parse_dates=['LocalTime'], dayfirst=False)
    if 'Wh' not in df.columns:
        raise SystemExit('No Wh column found in CSV')

    df = df[['LocalTime', 'Wh']].dropna(subset=['LocalTime'])
    df = df.set_index('LocalTime').sort_index()

    # ensure daily frequency (reindex if necessary)
    last = df.index.max()
    first = df.index.min()
    full_idx = pd.date_range(start=first, end=last, freq='D')
    df = df.reindex(full_idx)

    # simple imputation for missing Wh values
    df['Wh'] = df['Wh'].astype(float)
    df['Wh'] = df['Wh'].ffill().bfill()

    try:
        from autots import AutoTS
    except Exception as e:
        raise SystemExit('autots is not installed. Please run: pip install -r requirements.txt')

    # We'll hold out the last 30 days as test to compute metrics
    horizon = 30
    if len(df) <= horizon:
        raise SystemExit('Not enough data to hold out a 30-day test set')

    train_df = df.iloc[:-horizon].copy()
    test_df = df.iloc[-horizon:].copy()

    model = AutoTS(forecast_length=horizon, frequency='D', ensemble='simple', model_list='superfast', n_jobs=1, max_generations=1)

    # AutoTS expects wide format
    train_wide = train_df[['Wh']]

    print('Fitting AutoTS on training set...')
    model = model.fit(train_wide)

    print('Generating prediction...')
    prediction = model.predict()
    forecast = prediction.forecast

    # set forecast index to the test dates
    try:
        forecast.index = test_df.index
    except Exception:
        pass

    out_csv = os.path.join(out_dir, 'forecast_Wh_autots_30d.csv')
    forecast.to_csv(out_csv, index=True)

    # Compute evaluation metrics comparing forecast vs test
    # forecast may have column "Wh" or similar; pick first column
    if 'Wh' in forecast.columns:
        y_pred = forecast['Wh'].values
    else:
        y_pred = forecast.iloc[:, 0].values

    y_true = test_df['Wh'].values

    # ensure numeric
    y_pred = y_pred.astype(float)
    y_true = y_true.astype(float)

    mae = mean_absolute_error(y_true, y_pred)
    # MASE: scale by mean absolute naive (lag-1) on training set
    denom = np.mean(np.abs(np.diff(train_df['Wh'].astype(float).values)))
    mase = mae / denom if denom != 0 else np.nan
    # SMAPE
    smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)) * 100
    # R^2
    r2 = r2_score(y_true, y_pred)

    scores = {
        'MAE': float(mae),
        'MASE': float(mase) if not np.isnan(mase) else None,
        'SMAPE(%)': float(smape),
        'R2': float(r2),
    }

    print('\nEvaluation on last {} days:'.format(horizon))
    for k, v in scores.items():
        print(f'{k}: {v}')

    # save metrics
    try:
        metrics_path = os.path.join(out_dir, 'forecast_Wh_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # save model results summary
    try:
        res_path = os.path.join(out_dir, 'autots_model_results.json')
        with open(res_path, 'w', encoding='utf-8') as f:
            json.dump(model.results(), f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    print('\nForecast saved to', out_csv)


if __name__ == '__main__':
    main()
