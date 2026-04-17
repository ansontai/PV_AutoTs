import os
import pandas as pd
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
from datetime import datetime

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import builtins

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


def export_model_rankings(model, out_dir, horizon):
    """Export model rankings to CSV and Markdown.

    Attempts to use model.results('validation') or model.results() and
    heuristically pick a score column to sort by. Writes two files into
    out_dir: model_full_rankings_{horizon}d.csv and .md.
    """
    try:
        # Try validation results first
        try:
            all_results = model.results('validation')
        except Exception:
            all_results = model.results()

        # If results is a DataFrame-like object
        if hasattr(all_results, 'columns'):
            df_all = all_results.copy()
            preferred = ['Validation Score', 'validation_score', 'Validation_Score', 'Validation MAE', 'Validation SMAPE', 'SMAPE(%)', 'Score', 'score', 'validation MAE']
            score_col = None
            for c in preferred:
                if c in df_all.columns:
                    score_col = c
                    break
            if score_col is None:
                numeric_cols = [c for c in df_all.columns if pd.api.types.is_numeric_dtype(df_all[c]) and c.lower() not in ('id', 'rank')]
                score_col = numeric_cols[0] if numeric_cols else None

            ascending = True
            if score_col is not None:
                sc_lower = score_col.lower()
                if any(tok in sc_lower for tok in ('r2', 'r_squared', 'r^2', 'score')):
                    ascending = False
                if any(tok in sc_lower for tok in ('mae', 'smape', 'error', 'mape', 'rmse', 'rmsse')):
                    ascending = True

            if score_col is not None:
                try:
                    df_sorted = df_all.sort_values(by=score_col, ascending=ascending).reset_index(drop=True)
                except Exception:
                    df_sorted = df_all.reset_index(drop=True)
            else:
                df_sorted = df_all.reset_index(drop=True)

            rank_csv_all = os.path.join(out_dir, f'model_full_rankings_{horizon}d.csv')
            df_sorted.to_csv(rank_csv_all, index_label='rank')

            rank_md_all = os.path.join(out_dir, f'model_full_rankings_{horizon}d.md')
            with open(rank_md_all, 'w', encoding='utf-8') as mf:
                mf.write(f'# Model Full Rankings ({horizon}d)\n\n')
                mf.write(f'Sorted by: {score_col or "(original order)"} (ascending={ascending})\\n\n')
                cols = ['rank'] + list(df_sorted.columns)
                mf.write('|' + '|'.join(cols) + '|\n')
                mf.write('|' + '|'.join(['-'] * len(cols)) + '|\n')
                for i, row in df_sorted.reset_index().iterrows():
                    cells = [str(i+1)] + [str(row.get(c, '')) for c in df_sorted.columns]
                    mf.write('|' + '|'.join(cells) + '|\n')

            return rank_csv_all, rank_md_all
        else:
            # Not tabular; cannot produce full rankings
            return None, None
    except Exception as e:
        raise

#
# 非數值標記（例如 X、空字串）轉 NaN 並轉數值型
# 基於 Wh 建立 lag (1,7,14,30) 與 rolling mean/std（7/14/30）
# 為數值 regressors 建 rolling-7 mean
# 加入日曆特徵（weekday、is_weekend、month、dayofyear、sin/cos）
# 保守填補（預設 ffill then bfill）
#
def prepare_features(df, target_col='Wh', lags=(1, 7, 14, 30), rolls=(7, 14, 30),
                     add_calendar=True, fill_method='ffill', add_lag365=False):
    """Prepare features: clean, numeric convert, add lags, rolling stats and calendar features.

    Args:
        df (pd.DataFrame): DataFrame with DatetimeIndex or a 'LocalTime' column.
        target_col (str): name of target column (e.g. 'Wh').
        lags (iterable): lag offsets to create for the target.
        rolls (iterable): rolling window sizes for mean/std features.
        add_calendar (bool): whether to add calendar features.
        fill_method (str): 'ffill' or 'bfill' or other for fillna behavior.
        add_lag365 (bool): whether to add lag-365 feature.

    Returns:
        pd.DataFrame: copy with added features.
    """
    df = df.copy()

    # ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'LocalTime' in df.columns:
            df.index = pd.to_datetime(df['LocalTime'], errors='coerce')
            df = df.sort_index()

    # normalize common non-numeric markers and convert to numeric where possible
    df = df.replace({'X': np.nan, '': np.nan})
    for c in df.columns:
        if c == 'LocalTime':
            continue
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # fill missing values conservatively
    if fill_method == 'ffill':
        df = df.ffill().bfill()
    elif fill_method == 'bfill':
        df = df.bfill().ffill()
    else:
        df = df.fillna(0)

    # target lags and rolling features
    if target_col in df.columns:
        for lag in lags:
            df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
        if add_lag365:
            df[f'{target_col}_lag365'] = df[target_col].shift(365)
        for w in rolls:
            df[f'{target_col}_roll_mean_{w}'] = df[target_col].shift(1).rolling(window=w, min_periods=1).mean()
            df[f'{target_col}_roll_std_{w}'] = df[target_col].shift(1).rolling(window=w, min_periods=1).std().fillna(0)

    # regressors rolling means (one example window)
    numeric_regs = [c for c in df.columns if c not in (target_col, 'LocalTime') and pd.api.types.is_numeric_dtype(df[c])]
    for col in numeric_regs:
        df[f'{col}_roll7'] = df[col].shift(1).rolling(7, min_periods=1).mean()

    # calendar features
    if add_calendar and isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        df['dayofweek'] = idx.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['month'] = idx.month
        df['dayofyear'] = idx.dayofyear
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)

    # drop rows that are completely empty
    df = df.dropna(axis=0, how='all')
    return df



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

    # === 特徵工程：加入 lag / rolling / calendar 等衍生欄位 ===
    try:
        value_df = prepare_features(value_df, target_col='Wh', lags=(1, 7, 14, 30), rolls=(7, 14, 30), add_lag365=False)
        print('Prepared features, dataframe shape:', value_df.shape)
    except Exception as e:
        print('Feature preparation failed, continuing with original value_df:', e)

    # Keep univariate Wh series for forecasting
    df = value_df[['Wh']].copy()
    df['Wh'] = df['Wh'].astype(float).ffill().bfill()

    # horizons to run
    #horizons = [30, 60, 90]
    #horizons = [9, 6, 3]
    #horizons = [30, 60, 90]
    horizons = [9, 6, 3]  ### 測試用短 horizon，快速驗證流程
    # horizons = [90, 60, 30]  ### 從長 horizon 開始訓練，讓後續短 horizon 可以嘗試匯入前一個長 horizon 的 template 來加速搜索
    # default model_list to use when no template or when falling back
    # Can be a preset like 'superfast', 'fast', or a list of model names
    # default_model_list = 'superfast'
    # default_model_list = 'default'
    # default_model_list = 'all'
    # Remove models that require explicit future_regressor (e.g., RollingRegression, GLS)
    # default_model_list = ['AverageValueNaive', 'LastValueNaive', 'Theta', 'UnobservedComponents', 'SeasonalNaive']
    default_model_list=[
                  # 'ETS', # 容易變成水平線的模型
                  'Theta', # 「幾乎不會是水平線」的安全模型
                  'ARIMA',
                  # 'Prophet', # 若沒給到夠的時間特徵或資料太短，可能變得偏平或偏保守
                  # 'GLM', # 若沒給到夠的時間特徵或資料太短，可能變得偏平或偏保守
                  'RollingRegression', # 第二優先（穩定、不平）
                  'WindowRegression', # 第二優先（穩定、不平）
                  'DatepartRegression', # 只要 frequency 合理（日 / 週 / 月）➡ 幾乎不會是水平線
                  # 'TBATS', # 進階（但一定不平）
                  # 'SeasonalNaive', # 最容易變成水平線的模型
                  # 'AverageValueNaive', # 最容易變成水平線的模型
                  # 'LastValueNaive', # 最容易變成水平線的模型
                  # 'UnobservedComponents', # 容易變成水平線的模型
                  ]


    try:
        from autots import AutoTS
    except Exception as e:
        raise SystemExit('autots is not installed. Please run: pip install -r requirements.txt')
    # If torch is referenced inside autots (some autots versions have a bug
    # referencing `torch` without importing), ensure a safe fallback so the
    # module-level reference doesn't raise NameError when torch isn't installed.

    template_path = None

    # prepare output naming and optional infinite-loop control
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    # Toggle infinite looping here
    # InfiniteLoop = True ### Set to True to enable infinite looping (e.g., for scheduled runs); set to False for single execution.
    InfiniteLoop = False

    # index persistence file (keeps running index across invocations)
    index_file = os.path.join(base, f'{script_name}_infinite_index.txt')
    try:
        if os.path.exists(index_file):
            with open(index_file, 'r', encoding='utf-8') as f:
                cur_index = int(f.read().strip() or '1')
        else:
            cur_index = 1
    except Exception:
        cur_index = 1

    while True:
        ts = datetime.now().strftime('%y%m%d_%H%M%S')
        out_root = os.path.normpath(os.path.join(base, 'output', f"{script_name}_{ts}-{cur_index}"))
        os.makedirs(out_root, exist_ok=True)
        # Reset template_path for this outer-loop run so the first horizon
        # in this loop always exports a template into this run's out_root.
        template_path = None

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
                # model_list=[
                #   # 'ETS', # 容易變成水平線的模型
                #   'Theta', # 「幾乎不會是水平線」的安全模型
                #   'ARIMA',
                #   'SARIMA',
                #   # 'Prophet', # 若沒給到夠的時間特徵或資料太短，可能變得偏平或偏保守
                #   # 'GLM', # 若沒給到夠的時間特徵或資料太短，可能變得偏平或偏保守
                #   'RollingRegression', # 第二優先（穩定、不平）
                #   'WindowRegression', # 第二優先（穩定、不平）
                #   'DatepartRegression', # 只要 frequency 合理（日 / 週 / 月）➡ 幾乎不會是水平線
                #   'TBATS', # 進階（但一定不平）
                #   # 'SeasonalNaive', # 最容易變成水平線的模型
                #   # 'AverageValueNaive', # 最容易變成水平線的模型
                #   # 'LastValueNaive', # 最容易變成水平線的模型
                #   # 'UnobservedComponents', # 容易變成水平線的模型
                #   ],
                # ensemble=[
                #   # 'auto',
                #   # 'weighted', # 平線模型「很吃香」
                #   # 'simple', # 平均 or 中位數  只要一半模型是平線 → 結果就是平線
                #   # 'horizontal', # 只要有一個模型（例如 Theta）不是平線，horizontal 就能「撐起」整體預測不被抹平成水平線
                #   "mosaic", # 如果你的問題是「後面都變平線」 → 用 mosaic 就對了！mosaic 會在模型表現不明顯時，傾向選擇「多樣化」的模型組合，避免被單一平線模型主導預測結果。當然，如果你的資料真的很適合平線模型，mosaic 也會選擇平線模型，但它不會讓平線模型「獨大」，因此能夠在很多情況下避免預測結果被抹平成水平線。
                #   ],
                model_list=default_model_list,
                transformer_list=[
                  # 'ClipOutliers',
                  # 'RobustScaler',
                  # 'SeasonalDifference'
                  "DifferencedTransformer", # 避免被「抹平」成水平線
                  "Scaler", # 避免被「抹平」成水平線
                  ],
                  # n_jobs=4,
                  n_jobs=4, ### 使用 4 個核心來加速 AutoTS 搜索；若要使用所有核心請設為 -1。
                max_generations=1, ### superFast 模式下，1 代就能找到不錯的模型了；如果你有時間和算力，可以增加 generations 數量來尋找更優模型，但也可能會增加過擬合風險，尤其是資料量較小時。
                max_generations=30,
                # max_generations=1,
                num_validations=2,
                min_allowed_train_percent=0.33,
                no_negatives=True,
                )

            # Instantiate AutoTS (template import disabled)
            safe_ats_kwargs = {k: v for k, v in ats_kwargs.items() if k not in ('forecast_length', 'model_list')}
            model = AutoTS(model_list=default_model_list, **safe_ats_kwargs)
            # ensure forecast_length set explicitly
            model.forecast_length = horizon

            # If we have an exported template from the first (longest) horizon,
            # try to import it for subsequent (shorter) horizons to speed up search.
            if template_path is not None and idx > 0:
                try:
                    imp = getattr(model, 'import_template', None)
                    if callable(imp):
                        imp(template_path)
                        print('Imported AutoTS template from', template_path)
                    else:
                        # best-effort fallback: some AutoTS versions expose import_template differently
                        try:
                            model.import_template(template_path)
                            print('Imported AutoTS template (fallback) from', template_path)
                        except Exception:
                            print('No import_template available on this AutoTS instance; continuing without template')
                except Exception as e:
                    print('Failed to import template, continuing without it:', e)

            train_wide = train_df[['Wh']]
            print('Fitting AutoTS on training set...')
            # AutoTS.fit does not accept `metric` on some versions; use default fit API.
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

            # Export best model scores (CSV) and a small PNG summary of numeric metrics
            try:
                try:
                    results_df = model.results('validation')
                except Exception:
                    results_df = model.results()
                if isinstance(results_df, str):
                    print('No model results available to export best-model scores')
                else:
                    # ensure best model id is available
                    best_id = None
                    try:
                        best_id = getattr(model, 'best_model_id', None)
                    except Exception:
                        best_id = None
                    if not best_id:
                        try:
                            model.parse_best_model()
                            best_id = getattr(model, 'best_model_id', None)
                        except Exception:
                            best_id = None

                    if best_id:
                        best_rows = results_df[results_df['ID'] == best_id]
                        # fallback to initial results if needed
                        if best_rows.empty and hasattr(model, 'initial_results'):
                            try:
                                best_rows = model.initial_results.model_results[
                                    model.initial_results.model_results['ID'] == best_id
                                ]
                            except Exception:
                                pass

                        if not best_rows.empty:
                            bm_csv = os.path.join(out_dir, f'autots_best_model_scores_{horizon}d.csv')
                            best_rows.to_csv(bm_csv, index=False)

                            # numeric summary for plotting
                            num_df = best_rows.select_dtypes(include=[np.number]).copy()
                            if not num_df.empty:
                                summary = num_df.mean(axis=0).sort_values()
                                bm_png = os.path.join(out_dir, f'autots_best_model_scores_{horizon}d.png')
                                plt.figure(figsize=(8, max(4, 0.3 * len(summary))))
                                summary.plot(kind='barh', color='#264653')
                                plt.title(f'Best Model Metrics Summary ({horizon}d)')
                                plt.xlabel('Value')
                                plt.tight_layout()
                                plt.savefig(bm_png, dpi=150)
                                plt.close()
                                print('Best model scores saved to', bm_csv, 'and', bm_png)
                            else:
                                print('Best model numeric metrics missing; CSV saved to', bm_csv)
                        else:
                            print('Best model rows not found in results for ID', best_id)
                    else:
                        print('Best model id unavailable')
            except Exception as e:
                print('Failed to export best-model scores:', e)

            # === 產生所有嘗試過模型的排名（易讀 CSV + Markdown） ===
            try:
                try:
                    rank_csv_all, rank_md_all = export_model_rankings(model, out_dir, horizon)
                    if rank_csv_all and rank_md_all:
                        print('Full model rankings saved to', rank_csv_all)
                        print('Readable full rankings saved to', rank_md_all)
                    else:
                        print('model.results() did not return a tabular structure; skipping full rankings export')
                except Exception as e:
                    print('Failed to export full model rankings:', e)
            except Exception as e:
                print('Failed to export full model rankings (outer):', e)

            # export template only for the first (longest) trained horizon if not yet exported
            try:
                if template_path is None:
                    template_dir = os.path.normpath(os.path.join(out_root, 'autoTs_template'))
                    os.makedirs(template_dir, exist_ok=True)
                    template_path = os.path.join(template_dir, f'autoTs_template_{horizon}d.csv')
                    model.export_template(template_path, models='best', n=1, max_per_model_class=1, include_results=True)
                    print('Best model template saved to', template_path)
                    # also save a JSON copy (useful for other programs) and an import example
                    try:
                        try:
                            tmp_df = pd.read_csv(template_path)
                            json_path = os.path.join(template_dir, f'autoTs_template_{horizon}d.json')
                            tmp_df.to_json(json_path, orient='records', force_ascii=False, indent=2)
                            print('Template JSON saved to', json_path)
                        except Exception:
                            # fallback: save CSV contents as a JSON field
                            with open(template_path, 'r', encoding='utf-8') as tf:
                                txt = tf.read()
                            json_path = os.path.join(template_dir, f'autoTs_template_{horizon}d.json')
                            with open(json_path, 'w', encoding='utf-8') as jf:
                                json.dump({'template_csv': txt}, jf, ensure_ascii=False, indent=2)
                            print('Template saved as JSON text to', json_path)
                    except Exception as e:
                        print('Failed to save template JSON copy:', e)

                    try:
                        import_example_path = os.path.join(template_dir, 'import_example.py')
                        with open(import_example_path, 'w', encoding='utf-8') as ef:
                            ef.write('from autots import AutoTS\n')
                            ef.write("# create AutoTS and import the template to reuse the saved best-model settings\n")
                            ef.write("model = AutoTS(model_list='default', n_jobs=1)\n")
                            # use raw string for Windows paths to avoid escape issues
                            ef.write(f"model.import_template(r'''{template_path}''')\n")
                            ef.write("# then call model.fit()/predict() as usual\n")
                        print('Import example saved to', import_example_path)
                    except Exception as e:
                        print('Failed to write import example:', e)
            except Exception as e:
                print('Failed to export best model template:', e)

            print('\nForecast saved to', out_csv)

        # increment and persist the run index
        try:
            cur_index = int(cur_index) + 1
            with open(index_file, 'w', encoding='utf-8') as f:
                f.write(str(cur_index))
        except Exception:
            pass

        if not InfiniteLoop:
            break


if __name__ == '__main__':
    main()
