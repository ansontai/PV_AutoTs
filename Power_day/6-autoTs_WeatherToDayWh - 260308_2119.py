import os
import pandas as pd
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
from datetime import datetime

matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_forecast_comparison(plot_path, index, y_true, y_pred, y_naive, title=None, figsize=(12, 6), dpi=150):
    """Save a simple Forecast vs Actual vs Naive-Lag1 line chart.

    Parameters passed in to avoid reliance on outer scope.
    """
    import matplotlib.dates as mdates
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
    """Save a legacy gray-styled Forecast vs Actual vs Naive-Lag1 chart.

    All required values should be passed in so the function does not rely on
    external scope. Metrics (mae, rmsse, smape) are optional and will be
    displayed if provided.
    """
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
    if mase is not None or rmsse is not None or smape is not None:
        metrics_text = f"MASE={mase:.3f}\nRMSSE={rmsse:.3f}\nSMAPE={smape:.2f}%" if (mase is not None and rmsse is not None and smape is not None) else ''
        if metrics_text:
            ax.text(1.03, 0.98, metrics_text, transform=ax.transAxes,
                    fontsize=10, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 0.60), fontsize=10, frameon=False)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def compute_forecast_scores(y_true, y_pred, train_series):
    """Compute common forecast scores and return a scores dict plus key metrics.

    Parameters:
    - y_true: array-like of true values
    - y_pred: array-like of predicted values
    - train_series: array-like of training series (used for denom in MASE/RMSSE)

    Returns:
    - scores: dict with MAE, MASE_lag1, RMSSE, SMAPE(%), R2
    - mae, rmsse, smape: numeric values for convenience (may be NaN)
    """
    import numpy as _np
    from sklearn.metrics import mean_absolute_error as _mae, r2_score as _r2

    y_true = _np.asarray(y_true, dtype=float)
    y_pred = _np.asarray(y_pred, dtype=float)
    train_vals = _np.asarray(train_series, dtype=float)

    mae = float(_mae(y_true, y_pred))
    # MASE using lag-1 mean absolute diff of training series
    denom = _np.mean(_np.abs(_np.diff(train_vals))) if train_vals.size > 1 else 0.0
    mase = float(mae / denom) if denom != 0 else _np.nan

    rmse = float(_np.sqrt(_np.mean((y_pred - y_true) ** 2)))
    denom_rmsse = float(_np.sqrt(_np.mean(_np.diff(train_vals) ** 2))) if train_vals.size > 1 else 0.0
    rmsse = float(rmse / denom_rmsse) if denom_rmsse != 0 else _np.nan

    smape = float(_np.mean(2.0 * _np.abs(y_pred - y_true) / (_np.abs(y_true) + _np.abs(y_pred) + 1e-9)) * 100)
    r2 = float(_r2(y_true, y_pred))

    scores = {
        'MAE': mae,
        'MASE_lag1': float(mase) if not _np.isnan(mase) else None,
        'RMSSE': float(rmsse) if not _np.isnan(rmsse) else None,
        'SMAPE(%)': smape,
        'R2': r2,
    }

    return scores, mae, mase, rmsse, smape


def compute_and_save_feature_weights(train_value_df, out_dir, horizon=None, top_n=None):
    """Compute correlations of features vs Wh, save CSV/JSON/top-N CSV and a horizontal bar PNG.

    Args:
        train_value_df (pd.DataFrame): training dataframe containing 'Wh' and feature columns
        out_dir (str): output directory to save files
        horizon (int|None): optional horizon used for naming
        top_n (int|None): number of top features to save/plot (defaults to horizon or 10)

    Returns: dict with saved file paths and the weights DataFrame
    """
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

    # paths
    weights_csv_path = os.path.join(out_dir, 'feature_weights_vs_Wh.csv')
    weights_json_path = os.path.join(out_dir, 'feature_weights_vs_Wh.json')
    weights_df.to_csv(weights_csv_path, index=False)
    with open(weights_json_path, 'w', encoding='utf-8') as f:
        json.dump(weights_df.to_dict(orient='records'), f, ensure_ascii=False, indent=2)

    # top-n
    if top_n is None:
        top_n = horizon if horizon is not None else min(10, len(weights_df))
    top_weights_df = weights_df.head(top_n).copy()
    top_weights_csv_path = os.path.join(out_dir, f'feature_weights_top{top_n}_vs_Wh.csv')
    top_weights_df.to_csv(top_weights_csv_path, index=False)
    top_weights_plot_path = os.path.join(out_dir, f'feature_weights_top{top_n}_vs_Wh.png')

    # plot
    plt.figure(figsize=(12, 7))
    plt.barh(top_weights_df['column'][::-1], top_weights_df['weight'][::-1], color='#2a9d8f')
    plt.title(f'Top {top_n} Feature Weights vs Wh')
    plt.xlabel('Normalized Weight')
    plt.ylabel('Feature')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(top_weights_plot_path, dpi=150)
    plt.close()

    return {
        'weights_csv': weights_csv_path,
        'weights_json': weights_json_path,
        'top_weights_csv': top_weights_csv_path,
        'top_weights_png': top_weights_plot_path,
        'weights_df': weights_df,
    }



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
    horizons = [9, 6, 3]

    try:
        from autots import AutoTS
    except Exception as e:
        raise SystemExit('autots is not installed. Please run: pip install -r requirements.txt')

    template_path = None

    # prepare timestamped output root: output/<script_name>_<yyMMDD_HHMM>/
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    ts = datetime.now().strftime('%y%m%d_%H%M')
    out_root = os.path.normpath(os.path.join(base, 'output', f"{script_name}_{ts}"))
    os.makedirs(out_root, exist_ok=True)

    for idx, horizon in enumerate(horizons):
        print(f'\n===== 預測未來 {horizon} 天，測試集長度 {horizon} =====')
        out_dir = os.path.normpath(os.path.join(out_root, f'{horizon}d'))
        os.makedirs(out_dir, exist_ok=True)

        if len(df) <= horizon:
            print(f'資料長度不足，無法進行 {horizon} 天預測，跳過...')
            continue

        train_df = df.iloc[:-horizon].copy()
        test_df = df.iloc[-horizon:].copy()
        train_value_df = value_df.iloc[:-horizon].copy()

        # Prepare AutoTS kwargs; if we have a template from a previous (longer) run,
        # try to use it to speed up model search.
        ats_kwargs = dict(
            forecast_length=horizon,
            frequency='D',
            ensemble=['auto', 'horizontal-max'],
            transformer_list=['ClipOutliers', 'Detrend', 'SeasonalDifference'],
            n_jobs=-1,
            max_generations=15,
            num_validations=3,
            no_negatives=True,
        )

        ### template import and outport
        if template_path is not None:
            # Prefer to pass template_file if AutoTS supports it; otherwise
            # fall back to extracting a model_list from the template CSV.
            try:
                model = AutoTS(template_file=template_path, **ats_kwargs)
            except TypeError:
                # Older/newer AutoTS may not accept template_file in constructor.
                # Try to parse template CSV to generate a constrained model_list.
                try:
                    temp_df = pd.read_csv(template_path)
                    model_col = None
                    for c in temp_df.columns:
                        if 'model' in c.lower():
                            model_col = c
                            break
                    if model_col is not None:
                        models_from_template = temp_df[model_col].dropna().astype(str).tolist()
                        # deduplicate while preserving order
                        seen = set()
                        constrained_models = []
                        for m in models_from_template:
                            if m not in seen:
                                seen.add(m)
                                constrained_models.append(m)
                        if len(constrained_models) > 0:
                            ats_kwargs['model_list'] = constrained_models[:10]
                        else:
                            ats_kwargs['model_list'] = 'superfast'
                    else:
                        ats_kwargs['model_list'] = 'superfast'
                except Exception:
                    ats_kwargs['model_list'] = 'superfast'
                model = AutoTS(**ats_kwargs)
        else:
            model = AutoTS(model_list='superfast', **{k: v for k, v in ats_kwargs.items() if k != 'forecast_length'})
            # ensure forecast_length set explicitly
            model.forecast_length = horizon

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

        # compute evaluation scores using helper (now returns MASE too)
        scores, mae, mase, rmsse, smape = compute_forecast_scores(y_true, y_pred, train_df['Wh'].astype(float).values)

        # feature weights (delegated to helper)
        ### compute_and_save_feature_weights
        try:
            weights_info = compute_and_save_feature_weights(train_value_df, out_dir, horizon=horizon, top_n=30)
            print('Top-weights chart saved to', weights_info.get('top_weights_png'))
        except Exception as e:
            print('Failed to compute/save feature weights:', e)

        print(f'\nEvaluation on last {horizon} days:')
        for k, v in scores.items():
            print(f'{k}: {v}')

        # plot forecast vs actual
        try:
            plot_path = os.path.join(out_dir, f'forecast_vs_actual_vs_naive_lag1_{horizon}d.png')
            plot_df = pd.DataFrame({'Actual': y_true, 'Forecast': y_pred, 'NaiveLag1': y_naive}, index=test_df.index)
            plot_forecast_comparison(
                plot_path,
                plot_df.index,
                plot_df['Actual'],
                plot_df['Forecast'],
                plot_df['NaiveLag1'],
                title=f'Wh Forecast vs Actual vs Naive Lag-1 ({horizon}d)',
                figsize=(12, 6),
                dpi=150,
            )
            print('Comparison chart saved to', plot_path)
        except Exception as e:
            print('Failed to generate comparison chart:', e)

        ### plot_forecast_comparison_legacy
        # Also save a gray-styled variant using the previous (legacy) format
        try:
            plot_path = os.path.join(out_dir, f'forecast_vs_actual_vs_naive_lag1_{horizon}d-format2.png')
            plot_df = pd.DataFrame({'Actual': y_true, 'Forecast': y_pred, 'NaiveLag1': y_naive}, index=test_df.index)
            plot_forecast_comparison_legacy(
                plot_path,
                plot_df.index,
                plot_df['Actual'],
                plot_df['Forecast'],
                plot_df['NaiveLag1'],
                mase=mase,
                rmsse=rmsse,
                smape=smape,
                title=f'Wh Forecast vs Actual vs Naive Lag-1 ({horizon}d)',
                figsize=(6, 3),
                dpi=300,
            )
            print('Legacy gray comparison chart saved to', plot_path)
        except Exception as e:
            print('Failed to generate legacy gray comparison chart:', e)

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

        # export template only for the first (longest) trained horizon if not yet exported
        try:
            if template_path is None:
                template_dir = os.path.normpath(os.path.join(out_root, 'autoTs_template'))
                os.makedirs(template_dir, exist_ok=True)
                template_path = os.path.join(template_dir, f'autoTs_template_{horizon}d.csv')
                model.export_template(template_path, models='best', n=1, max_per_model_class=1, include_results=True)
                print('Best model template saved to', template_path)
        except Exception as e:
            print('Failed to export best model template:', e)

        print('\nForecast saved to', out_csv)

    # # plot forecast vs actual vs naive-lag1 baseline (30d paper-style)
    # try:
    #     plot_path = os.path.join(out_dir, 'forecast_vs_actual_vs_naive_lag1_gray.png')
    #     plot_df = pd.DataFrame(
    #         {
    #             'Actual': y_true,
    #             'Forecast': y_pred,
    #             'NaiveLag1': y_naive,
    #         },
    #         index=test_df.index,
    #     )
    #     import matplotlib.dates as mdates
    #     plt.figure(figsize=(6, 3), dpi=300)
    #     plt.plot(plot_df.index, plot_df['Actual'], label='Actual', color='black', linewidth=2.5)
    #     plt.plot(plot_df.index, plot_df['Forecast'], label='AutoTS Forecast', color='dimgray', linewidth=2.5)
    #     plt.plot(plot_df.index, plot_df['NaiveLag1'], label='Naive Lag-1', color='gray', linewidth=2, linestyle='--')
    #     plt.title('Wh Forecast vs Actual vs Naive Lag-1 (30d)', fontsize=15, pad=12)
    #     plt.xlabel('Date', fontsize=13)
    #     plt.ylabel('Wh', fontsize=13)
    #     plt.grid(alpha=0.4, linestyle=':', linewidth=0.8)
    #     plt.xticks(fontsize=11, rotation=30)
    #     plt.yticks(fontsize=11)
    #     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    #     plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    #     metrics_text = f"MAE={mae:.2f}\nRMSSE={rmsse:.3f}\nSMAPE={smape:.2f}%"
    #     plt.gca().text(1.03, 0.98, metrics_text, transform=plt.gca().transAxes,
    #                fontsize=10, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
    #     plt.legend(loc='upper left', bbox_to_anchor=(1.01, 0.60), fontsize=10, frameon=False)
    #     plt.tight_layout(rect=[0, 0, 0.85, 1])
    #     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    #     plt.close()
    #     print('論文級 Comparison chart saved to', plot_path)
    # except Exception as e:
    #     print('Failed to generate comparison chart:', e)

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
