import os
import pandas as pd
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    # === 設定檔案路徑 ===
    base = os.path.dirname(__file__)
    csv_path = os.path.normpath(os.path.join(base, '..', 'csv', 'SolarRecord(260204)_d_forWh_WithCodis.csv'))

    # === 讀取每日發電資料 ===
    raw_df = pd.read_csv(csv_path, parse_dates=['LocalTime'], dayfirst=False)
    if 'Wh' not in raw_df.columns:
        raise SystemExit('No Wh column found in CSV')

    # === 時間序列前處理：補齊日期、數值型轉換、補值 ===
    raw_df = raw_df.dropna(subset=['LocalTime'])
    raw_df = raw_df.set_index('LocalTime').sort_index()
    first = raw_df.index.min()
    last = raw_df.index.max()
    full_idx = pd.date_range(start=first, end=last, freq='D')
    raw_df = raw_df.reindex(full_idx)

    # Convert all non-datetime columns to numeric if possible for weighting.
    value_df = raw_df.copy()
    for col in value_df.columns:
        value_df[col] = pd.to_numeric(value_df[col], errors='coerce')
    value_df = value_df.ffill().bfill()

    # Keep univariate Wh series for forecasting
    df = value_df[['Wh']].copy()
    df['Wh'] = df['Wh'].astype(float).ffill().bfill()

    # horizons to run
    #horizons = [30, 60, 90]
    horizons = [3, 6, 9]

    try:
        from autots import AutoTS
    except Exception as e:
        raise SystemExit('autots is not installed. Please run: pip install -r requirements.txt')

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

        model = AutoTS(
            forecast_length=horizon,
            frequency='D',
            ensemble=['auto', 'horizontal-max'],
            model_list='superfast',
            transformer_list=['ClipOutliers', 'Detrend', 'SeasonalDifference'],
            n_jobs=-1,
            max_generations=1,
            num_validations=2,
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

        # y_pred / y_true / naive-lag1
        if 'Wh' in forecast.columns:
            y_pred = forecast['Wh'].values
        else:
            y_pred = forecast.iloc[:, 0].values
        y_true = test_df['Wh'].values
        train_wh = train_df['Wh'].astype(float).values
        y_naive = np.r_[train_wh[-1], y_true[:-1]]

        y_pred = y_pred.astype(float)
        y_true = y_true.astype(float)

        mae = mean_absolute_error(y_true, y_pred)
        denom = np.mean(np.abs(np.diff(train_df['Wh'].astype(float).values)))
        mase = mae / denom if denom != 0 else np.nan
        # (removed MASE_lag365 calculation)
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        denom_rmsse = np.sqrt(np.mean(np.diff(train_df['Wh'].astype(float).values) ** 2))
        rmsse = rmse / denom_rmsse if denom_rmsse != 0 else np.nan
        smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)) * 100
        r2 = r2_score(y_true, y_pred)

        scores = {
            'MAE': float(mae),
            'MASE_lag1': float(mase) if not np.isnan(mase) else None,
            'RMSSE': float(rmsse) if not np.isnan(rmsse) else None,
            'SMAPE(%)': float(smape),
            'R2': float(r2),
        }

        # feature weights
        try:
            feature_cols = [c for c in train_value_df.columns if c != 'Wh']
            corr = train_value_df[feature_cols].corrwith(train_value_df['Wh'])
            corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            abs_corr = corr.abs()
            total_abs_corr = float(abs_corr.sum())
            weights = abs_corr / total_abs_corr if total_abs_corr > 0 else abs_corr * 0.0
            weights_df = pd.DataFrame({
                'column': corr.index,
                'corr_with_Wh': corr.values,
                'abs_corr': abs_corr.values,
                'weight': weights.values,
            }).sort_values('weight', ascending=False)

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
            print('Top-weights chart saved to', top_weights_plot_path)
        except Exception as e:
            print('Failed to compute/save feature weights:', e)

        print(f'\nEvaluation on last {horizon} days:')
        for k, v in scores.items():
            print(f'{k}: {v}')

        # plot forecast vs actual
        try:
            plot_path = os.path.join(out_dir, f'forecast_vs_actual_vs_naive_lag1_{horizon}d.png')
            plot_df = pd.DataFrame({'Actual': y_true, 'Forecast': y_pred, 'NaiveLag1': y_naive}, index=test_df.index)
            import matplotlib.dates as mdates
            if horizon == 30:
                plt.figure(figsize=(6, 3), dpi=300)
                plt.plot(plot_df.index, plot_df['Actual'], label='Actual', color='black', linewidth=2.5)
                plt.plot(plot_df.index, plot_df['Forecast'], label='AutoTS Forecast', color='dimgray', linewidth=2.5)
                plt.plot(plot_df.index, plot_df['NaiveLag1'], label='Naive Lag-1', color='gray', linewidth=2, linestyle='--')
                plt.title(f'Wh Forecast vs Actual vs Naive Lag-1 (30d)', fontsize=15, pad=12)
                plt.xlabel('Date', fontsize=13)
                plt.ylabel('Wh', fontsize=13)
                plt.grid(alpha=0.4, linestyle=':', linewidth=0.8)
                plt.xticks(fontsize=11, rotation=30)
                plt.yticks(fontsize=11)
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
                metrics_text = f"MAE={mae:.2f}\nRMSSE={rmsse:.3f}\nSMAPE={smape:.2f}%"
                plt.gca().text(1.01, 0.98, metrics_text, transform=plt.gca().transAxes, fontsize=10, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
                plt.legend(loc='upper left', bbox_to_anchor=(1.01, 0.60), fontsize=10, frameon=False)
                plt.tight_layout(rect=[0, 0, 0.85, 1])
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print('論文級 Comparison chart saved to', plot_path)
            else:
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

        # (已移除 lag-365 比較圖以簡化輸出與視覺化)

        # save metrics and model results
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

        # export template
        try:
            template_dir = os.path.normpath(os.path.join(base, '..', 'output', 'autoTs_template'))
            os.makedirs(template_dir, exist_ok=True)
            template_path = os.path.join(template_dir, f'autoTs_template_{horizon}d.csv')
            model.export_template(template_path, models='best', n=1, max_per_model_class=1, include_results=True)
            print('Best model template saved to', template_path)
        except Exception as e:
            print('Failed to export best model template:', e)

        print('\nForecast saved to', out_csv)

    # plot forecast vs actual vs naive-lag1 baseline (30d paper-style)
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
        import matplotlib.dates as mdates
        plt.figure(figsize=(6, 3), dpi=300)
        plt.plot(plot_df.index, plot_df['Actual'], label='Actual', color='black', linewidth=2.5)
        plt.plot(plot_df.index, plot_df['Forecast'], label='AutoTS Forecast', color='dimgray', linewidth=2.5)
        plt.plot(plot_df.index, plot_df['NaiveLag1'], label='Naive Lag-1', color='gray', linewidth=2, linestyle='--')
        plt.title('Wh Forecast vs Actual vs Naive Lag-1 (30d)', fontsize=15, pad=12)
        plt.xlabel('Date', fontsize=13)
        plt.ylabel('Wh', fontsize=13)
        plt.grid(alpha=0.4, linestyle=':', linewidth=0.8)
        plt.xticks(fontsize=11, rotation=30)
        plt.yticks(fontsize=11)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        metrics_text = f"MAE={mae:.2f}\nRMSSE={rmsse:.3f}\nSMAPE={smape:.2f}%"
        plt.gca().text(1.01, 0.98, metrics_text, transform=plt.gca().transAxes,
                   fontsize=10, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
        plt.legend(loc='upper left', bbox_to_anchor=(1.01, 0.60), fontsize=10, frameon=False)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print('論文級 Comparison chart saved to', plot_path)
    except Exception as e:
        print('Failed to generate comparison chart:', e)

    # (已移除 lag-365 比較圖以簡化輸出與視覺化)

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
