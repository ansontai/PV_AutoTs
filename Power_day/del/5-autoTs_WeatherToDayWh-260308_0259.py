import os
import pandas as pd
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    base = os.path.dirname(__file__)
    csv_path = os.path.normpath(os.path.join(base, '..', 'csv', 'SolarRecord(260204)_d_forWh_WithCodis.csv'))
    out_dir = os.path.normpath(os.path.join(base, '..', 'output', '30d'))
    os.makedirs(out_dir, exist_ok=True)

    ## 讀取每日發電資料
    raw_df = pd.read_csv(csv_path, parse_dates=['LocalTime'], dayfirst=False)
    if 'Wh' not in raw_df.columns:
        raise SystemExit('No Wh column found in CSV')

    #
    # 做時間序列前處理
    # Power_day/autoTs_Weather_Wh.py:17 到 Power_day/autoTs_Weather_Wh.py:28 會：
    # 以日期當索引並排序
    # 補齊每天的日期索引（缺日補進來）
    # Wh 缺值用前後值補齊（ffill + bfill）
    #
    raw_df = raw_df.dropna(subset=['LocalTime'])
    raw_df = raw_df.set_index('LocalTime').sort_index()

    # ensure daily frequency (reindex if necessary)
    last = raw_df.index.max()
    first = raw_df.index.min()
    full_idx = pd.date_range(start=first, end=last, freq='D')
    raw_df = raw_df.reindex(full_idx)

    # Convert all non-datetime columns to numeric if possible for weighting.
    value_df = raw_df.copy()
    for col in value_df.columns:
        value_df[col] = pd.to_numeric(value_df[col], errors='coerce')
    value_df = value_df.ffill().bfill()

    # Keep univariate Wh series for current AutoTS forecast setup.
    df = value_df[['Wh']].copy()

    # simple imputation for missing Wh values
    df['Wh'] = df['Wh'].astype(float)
    df['Wh'] = df['Wh'].ffill().bfill()

    try:
        from autots import AutoTS
    except Exception as e:
        raise SystemExit('autots is not installed. Please run: pip install -r requirements.txt')

    ###
    ### 用 AutoTS 訓練與預測 30 天
    ###Power_day/autoTs_Weather_Wh.py:36 設 horizon = 30，最後 30 天當測試集，其餘當訓練集（Power_day/autoTs_Weather_Wh.  py:40）。
    ### Power_day/autoTs_Weather_Wh.py:43 用 AutoTS（model_list='superfast'）訓練並產生未來 30 天預測（Power_day/autoTs_Weather_Wh.py:52）。
    ### AutoTS 預測結果會存到 forecast_Wh_autots_30d.csv（Power_day/autoTs_Weather_Wh.py:56）。最後會計算 MAE、MASE、SMAPE 和 R^2 四個評估指標，並輸出到 console 和 forecast_Wh_metrics.json（Power_day/autoTs_Weather_Wh.py:63-80）。
    
    # We'll hold out the last 30 days as test to compute metrics
    horizon = 30
    if len(df) <= horizon:
        raise SystemExit('Not enough data to hold out a 30-day test set')

    train_df = df.iloc[:-horizon].copy()
    test_df = df.iloc[-horizon:].copy()
    train_value_df = value_df.iloc[:-horizon].copy()

    model = AutoTS(
      forecast_length=horizon, 
      frequency='D', 
      #ensemble='simple', 
      #ensemble='auto',
      ensemble=['weighted', 'horizontal'],
      #model_list='superfast',
      model_list='default',
      transformer_list=[
        'ClipOutliers',
        'Detrend',
        'SeasonalDifference'
        ],
      #n_jobs=1, 
      n_jobs=-2,
      max_generations=15,
      num_validations=2,
      no_negatives=True,
      )
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

    ## 輸出預測檔, 預測結果會存到：forecast_Wh_autots_30d.csv
    out_csv = os.path.join(out_dir, 'forecast_Wh_autots_30d.csv')
    forecast.to_csv(out_csv, index=True)

    # Compute evaluation metrics comparing forecast vs test
    # forecast may have column "Wh" or similar; pick first column
    if 'Wh' in forecast.columns:
        y_pred = forecast['Wh'].values
    else:
        y_pred = forecast.iloc[:, 0].values

    y_true = test_df['Wh'].values
    naive_mean = float(train_df['Wh'].astype(float).mean())
    y_naive = np.full(shape=len(y_true), fill_value=naive_mean, dtype=float)

    # ensure numeric
    y_pred = y_pred.astype(float)
    y_true = y_true.astype(float)

    ## 計算評估指標, 把預測值和最後 30 天真值比較，計算：
    mae = mean_absolute_error(y_true, y_pred)
    # MASE: scale by mean absolute naive (lag-1) on training set
    denom = np.mean(np.abs(np.diff(train_df['Wh'].astype(float).values)))
    mase = mae / denom if denom != 0 else np.nan
    # RMSSE: scale RMSE by RMS of naive (lag-1) errors on training set
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    denom_rmsse = np.sqrt(np.mean(np.diff(train_df['Wh'].astype(float).values) ** 2))
    rmsse = rmse / denom_rmsse if denom_rmsse != 0 else np.nan
    # SMAPE
    smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)) * 100
    # R^2
    r2 = r2_score(y_true, y_pred)

    scores = {
        'MAE': float(mae),
        'MASE': float(mase) if not np.isnan(mase) else None,
        'RMSSE': float(rmsse) if not np.isnan(rmsse) else None,
        'SMAPE(%)': float(smape),
        'R2': float(r2),
    }

    # Compute per-column weights by normalized absolute correlation to Wh.
    try:
        feature_cols = [c for c in train_value_df.columns if c != 'Wh']
        corr = train_value_df[feature_cols].corrwith(train_value_df['Wh'])
        corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        abs_corr = corr.abs()
        total_abs_corr = float(abs_corr.sum())
        if total_abs_corr > 0:
            weights = abs_corr / total_abs_corr
        else:
            weights = abs_corr * 0.0

        weights_df = pd.DataFrame(
            {
                'column': corr.index,
                'corr_with_Wh': corr.values,
                'abs_corr': abs_corr.values,
                'weight': weights.values,
            }
        ).sort_values('weight', ascending=False)

        weights_csv_path = os.path.join(out_dir, 'feature_weights_vs_Wh.csv')
        weights_json_path = os.path.join(out_dir, 'feature_weights_vs_Wh.json')
        weights_df.to_csv(weights_csv_path, index=False)
        with open(weights_json_path, 'w', encoding='utf-8') as f:
            json.dump(weights_df.to_dict(orient='records'), f, ensure_ascii=False, indent=2)

        # Export top-30 features as a separate file for quick inspection.
        top_n = horizon # 30
        top_weights_df = weights_df.head(top_n).copy()
        top_weights_csv_path = os.path.join(out_dir, 'feature_weights_top30_vs_Wh.csv')
        top_weights_df.to_csv(top_weights_csv_path, index=False)

        # Plot top-30 feature weights.
        top_weights_plot_path = os.path.join(out_dir, 'feature_weights_top30_vs_Wh.png')
        plt.figure(figsize=(12, 7))
        plt.barh(top_weights_df['column'][::-1], top_weights_df['weight'][::-1], color='#2a9d8f')
        plt.title('Top 30 Feature Weights vs Wh')
        plt.xlabel('Normalized Weight')
        plt.ylabel('Feature')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(top_weights_plot_path, dpi=150)
        plt.close()

        print('Feature weights saved to', weights_csv_path)
        print('Top-30 weights saved to', top_weights_csv_path)
        print('Top-30 weights chart saved to', top_weights_plot_path)
    except Exception as e:
        print('Failed to compute/save feature weights:', e)

    print('\nEvaluation on last {} days:'.format(horizon))
    for k, v in scores.items():
        print(f'{k}: {v}')

    # plot forecast vs actual vs naive-mean baseline
    try:
        plot_path = os.path.join(out_dir, 'forecast_vs_actual_vs_naive_mean.png')
        plot_df = pd.DataFrame(
            {
                'Actual': y_true,
                'Forecast': y_pred,
                'NaiveMean': y_naive,
            },
            index=test_df.index,
        )

        plt.figure(figsize=(12, 6))
        plt.plot(plot_df.index, plot_df['Actual'], label='Actual', linewidth=2)
        plt.plot(plot_df.index, plot_df['Forecast'], label='AutoTS Forecast', linewidth=2)
        plt.plot(plot_df.index, plot_df['NaiveMean'], label='Naive Mean', linewidth=2, linestyle='--')
        plt.title('Wh Forecast vs Actual vs Naive Mean')
        plt.xlabel('Date')
        plt.ylabel('Wh')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print('Comparison chart saved to', plot_path)
    except Exception as e:
        print('Failed to generate comparison chart:', e)

    # save metrics
    try:
        ## 輸出評估與模型資訊
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
