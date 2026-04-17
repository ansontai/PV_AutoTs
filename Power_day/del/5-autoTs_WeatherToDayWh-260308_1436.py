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

    ## 讀取每日發電資料
    raw_df = pd.read_csv(csv_path, parse_dates=['LocalTime'], dayfirst=False)
    if 'Wh' not in raw_df.columns:
        raise SystemExit('No Wh column found in CSV')

    # 做時間序列前處理
    raw_df = raw_df.dropna(subset=['LocalTime'])
    raw_df = raw_df.set_index('LocalTime').sort_index()
    last = raw_df.index.max()
    first = raw_df.index.min()
    full_idx = pd.date_range(start=first, end=last, freq='D')
    raw_df = raw_df.reindex(full_idx)
    value_df = raw_df.copy()
    for col in value_df.columns:
        value_df[col] = pd.to_numeric(value_df[col], errors='coerce')
    value_df = value_df.ffill().bfill()
    df = value_df[['Wh']].copy()
    df['Wh'] = df['Wh'].astype(float)
    df['Wh'] = df['Wh'].ffill().bfill()

    horizons = [30, 60, 90]
    #horizons = [30]
    for horizon in horizons:
        print(f'\n===== 預測未來 {horizon} 天，測試集長度 {horizon} =====')
        out_dir = os.path.normpath(os.path.join(base, '..', 'output', f'{horizon}d'))
        os.makedirs(out_dir, exist_ok=True)

        if len(df) <= horizon:
            print(f'資料長度不足，無法進行 {horizon} 天預測，跳過...')
            continue

        train_df = df.iloc[:-horizon].copy()
        test_df = df.iloc[-horizon:].copy()
        train_value_df = value_df.iloc[:-horizon].copy()

        try:
            from autots import AutoTS
        except Exception as e:
            raise SystemExit('autots is not installed. Please run: pip install -r requirements.txt')

        model = AutoTS(
          forecast_length=horizon, 
          frequency='D', 
          ensemble=['simple', 'weighted', 'horizontal-max'],
          model_list=[
              "ARIMA", # 數學時間序列
              "ETS",   # 平滑模型
              "GLM",
              "SeasonalNaive",
              "UnobservedComponents",
              "WindowRegression",  # ML學習過去模式
          ],
          transformer_list=[
              'ClipOutliers',      # 移除異常值
              'Detrend',           # 去除長期趨勢
              'SeasonalDifference' # 季節性差分
          ],
          n_jobs=-1,
          #max_generations=6,
          max_generations=60,
          num_validations=3,
          validation_method="backwards",
          no_negatives=True,
        )
        train_wide = train_df[['Wh']]
        print('Fitting AutoTS on training set...')
        model = model.fit(train_wide)
        print('Generating prediction...')
        prediction = model.predict()
        forecast = prediction.forecast
        try:
            forecast.index = test_df.index
        except Exception:
            pass
        out_csv = os.path.join(out_dir, f'forecast_Wh_autots_{horizon}d.csv')
        forecast.to_csv(out_csv, index=True)
        if 'Wh' in forecast.columns:
            y_pred = forecast['Wh'].values
        else:
            y_pred = forecast.iloc[:, 0].values
        y_true = test_df['Wh'].values
        # Use lag-1 naive forecast baseline (same naive concept as MASE scaling)
        train_wh = train_df['Wh'].astype(float).values
        y_naive = np.r_[train_wh[-1], y_true[:-1]]
        y_pred = y_pred.astype(float)
        y_true = y_true.astype(float)
        mae = mean_absolute_error(y_true, y_pred)
        denom = np.mean(np.abs(np.diff(train_df['Wh'].astype(float).values)))
        mase = mae / denom if denom != 0 else np.nan
        # MASE with seasonal lag-365 scaling (daily seasonality by year)
        train_wh_float = train_df['Wh'].astype(float).values
        if len(train_wh_float) > 365:
            denom_lag365 = np.mean(np.abs(train_wh_float[365:] - train_wh_float[:-365]))
            mase_lag365 = mae / denom_lag365 if denom_lag365 != 0 else np.nan
        else:
            mase_lag365 = np.nan
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        denom_rmsse = np.sqrt(np.mean(np.diff(train_df['Wh'].astype(float).values) ** 2))
        rmsse = rmse / denom_rmsse if denom_rmsse != 0 else np.nan
        smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)) * 100
        r2 = r2_score(y_true, y_pred)
        scores = {
            'MAE': float(mae),
            'MASE_lag1': float(mase) if not np.isnan(mase) else None,
            ##
            ## 計算方式是用訓練集 lag=365 的季節性 naive 分母：
            ##   mean(abs(y_t - y_{t-365}))
            ##   若訓練資料不足 366 筆或分母為 0，則回傳 None（原始為 np.nan）
            ## MASE_lag365 的概念是評估模型在捕捉年季節性方面的表現，對於具有明顯年季節性模式的資料特別有意義。
            ##   注意：在某些資料集上，尤其是訓練集較短或季節性不明顯的情況下，MASE_lag365 可能無法計算或解釋，因此在分析結果時應該考慮這些因素。
            ##
            'MASE_lag365': float(mase_lag365) if not np.isnan(mase_lag365) else None,
            'RMSSE': float(rmsse) if not np.isnan(rmsse) else None,
            'SMAPE(%)': float(smape),
            'R2': float(r2),
        }
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
            top_n = horizon
            top_weights_df = weights_df.head(top_n).copy()
            top_weights_csv_path = os.path.join(out_dir, f'feature_weights_top{horizon}_vs_Wh.csv')
            top_weights_df.to_csv(top_weights_csv_path, index=False)
            top_weights_plot_path = os.path.join(out_dir, f'feature_weights_top{horizon}_vs_Wh.png')
            plt.figure(figsize=(12, 7))
            plt.barh(top_weights_df['column'][::-1], top_weights_df['weight'][::-1], color='#2a9d8f')
            plt.title(f'Top {horizon} Feature Weights vs Wh')
            plt.xlabel('Normalized Weight')
            plt.ylabel('Feature')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(top_weights_plot_path, dpi=150)
            plt.close()
            print('Feature weights saved to', weights_csv_path)
            print(f'Top-{horizon} weights saved to', top_weights_csv_path)
            print(f'Top-{horizon} weights chart saved to', top_weights_plot_path)
        except Exception as e:
            print('Failed to compute/save feature weights:', e)
        print(f'\nEvaluation on last {horizon} days:')
        for k, v in scores.items():
            print(f'{k}: {v}')
        try:
            plot_path = os.path.join(out_dir, f'forecast_vs_actual_vs_naive_lag1_{horizon}d.png')
            plot_df = pd.DataFrame(
                {
                    'Actual': y_true,
                    'Forecast': y_pred,
                    'NaiveLag1': y_naive,
                },
                index=test_df.index,
            )
            plt.figure(figsize=(12, 6))
            plt.plot(plot_df.index, plot_df['Actual'], label='Actual', linewidth=2)
            plt.plot(plot_df.index, plot_df['Forecast'], label='AutoTS Forecast', linewidth=2)
            plt.plot(plot_df.index, plot_df['NaiveLag1'], label='Naive Lag-1', linewidth=2, linestyle='--')
            plt.title(f'Wh Forecast vs Actual vs Naive Lag-1 ({horizon}d)')
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

        # Additional chart against lag-365 naive baseline.
        try:
            lag365_plot_path = os.path.join(out_dir, 'forecast_vs_actual_vs_naive_lag365.png')
            naive_lag365 = df['Wh'].astype(float).shift(365).iloc[-horizon:].values
            lag365_plot_df = pd.DataFrame(
                {
                    'Actual': y_true,
                    'Forecast': y_pred,
                    'NaiveLag365': naive_lag365,
                },
                index=test_df.index,
            )

            plt.figure(figsize=(12, 6))
            plt.plot(lag365_plot_df.index, lag365_plot_df['Actual'], label='Actual', linewidth=2)
            plt.plot(lag365_plot_df.index, lag365_plot_df['Forecast'], label='AutoTS Forecast', linewidth=2)
            plt.plot(lag365_plot_df.index, lag365_plot_df['NaiveLag365'], label='Naive Lag-365', linewidth=2, linestyle='--')
            plt.title(f'Wh Forecast vs Actual vs Naive Lag-365 ({horizon}d)')
            plt.xlabel('Date')
            plt.ylabel('Wh')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(lag365_plot_path, dpi=150)
            plt.close()
            print('Lag-365 comparison chart saved to', lag365_plot_path)
        except Exception as e:
            print('Failed to generate lag-365 comparison chart:', e)
        try:
            metrics_path = os.path.join(out_dir, f'forecast_Wh_metrics_{horizon}d.json')
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(scores, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        try:
            res_path = os.path.join(out_dir, f'autots_model_results_{horizon}d.json')
            with open(res_path, 'w', encoding='utf-8') as f:
                json.dump(model.results(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        print(f'\nForecast saved to', out_csv)

        # 儲存訓練完成的最佳模型 template 到 output/autoTs_template
        try:
            template_dir = os.path.normpath(os.path.join(base, '..', 'output', 'autoTs_template'))
            os.makedirs(template_dir, exist_ok=True)
            template_path = os.path.join(template_dir, f'autoTs_template_{horizon}d.csv')
            model.export_template(
                template_path,
                models="best",
                n=1,
                max_per_model_class=1,
                include_results=True
            )
            print('Best model template saved to', template_path)
        except Exception as e:
            print('Failed to export best model template:', e)

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
    # Use lag-1 naive forecast baseline (same naive concept as MASE scaling)
    train_wh = train_df['Wh'].astype(float).values
    y_naive = np.r_[train_wh[-1], y_true[:-1]]

    # ensure numeric
    y_pred = y_pred.astype(float)
    y_true = y_true.astype(float)

    ## 計算評估指標, 把預測值和最後 30 天真值比較，計算：
    mae = mean_absolute_error(y_true, y_pred)
    # MASE: scale by mean absolute naive (lag-1) on training set
    denom = np.mean(np.abs(np.diff(train_df['Wh'].astype(float).values)))
    mase = mae / denom if denom != 0 else np.nan
    # MASE lag-365: scale by mean absolute seasonal naive (lag-365) on training set
    train_wh_float = train_df['Wh'].astype(float).values
    if len(train_wh_float) > 365:
        denom_lag365 = np.mean(np.abs(train_wh_float[365:] - train_wh_float[:-365]))
        mase_lag365 = mae / denom_lag365 if denom_lag365 != 0 else np.nan
    else:
        mase_lag365 = np.nan
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
        #'MASE': float(mase) if not np.isnan(mase) else None,
        'MASE_lag1': float(mase) if not np.isnan(mase) else None,
        # 'MASE_lag365': float(mase_lag365) if not np.isnan(mase_lag365) else None, ### data set 不足一年，無法計算季節性 MASE，回傳 None
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
        plot_path = os.path.join(out_dir, 'forecast_vs_actual_vs_naive_lag1.png')
        plot_df = pd.DataFrame(
            {
                'Actual': y_true,
                'Forecast': y_pred,
                'NaiveLag1': y_naive,
            },
            index=test_df.index,
        )

        plt.figure(figsize=(12, 6))
        plt.plot(plot_df.index, plot_df['Actual'], label='Actual', linewidth=2)
        plt.plot(plot_df.index, plot_df['Forecast'], label='AutoTS Forecast', linewidth=2)
        plt.plot(plot_df.index, plot_df['NaiveLag1'], label='Naive Lag-1', linewidth=2, linestyle='--')
        plt.title('Wh Forecast vs Actual vs Naive Lag-1')
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

    # plot forecast vs actual vs naive-lag365 baseline
    try:
        plot_path_lag365 = os.path.join(out_dir, 'forecast_vs_actual_vs_naive_lag365.png')
        y_naive_lag365 = df['Wh'].astype(float).shift(365).iloc[-horizon:].values
        plot_df_lag365 = pd.DataFrame(
            {
                'Actual': y_true,
                'Forecast': y_pred,
                'NaiveLag365': y_naive_lag365,
            },
            index=test_df.index,
        )

        plt.figure(figsize=(12, 6))
        plt.plot(plot_df_lag365.index, plot_df_lag365['Actual'], label='Actual', linewidth=2)
        plt.plot(plot_df_lag365.index, plot_df_lag365['Forecast'], label='AutoTS Forecast', linewidth=2)
        plt.plot(plot_df_lag365.index, plot_df_lag365['NaiveLag365'], label='Naive Lag-365', linewidth=2, linestyle='--')
        plt.title('Wh Forecast vs Actual vs Naive Lag-365')
        plt.xlabel('Date')
        plt.ylabel('Wh')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path_lag365, dpi=150)
        plt.close()
        print('Lag-365 comparison chart saved to', plot_path_lag365)
    except Exception as e:
        print('Failed to generate lag-365 comparison chart:', e)

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
