#!/usr/bin/env python3
"""AutoTS wrapper — 只使用 Prophet 模型並輸出 90 天預測。

此檔會找專案中的 Wh 時序（或第一個數值欄位），使用 AutoTS 並強制 model_list=['Prophet']。
輸出檔案：`prophet_autots_forecast_90d.csv` 至本檔同目錄。
"""
from __future__ import annotations
import os
import sys
import json
import platform
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import re
import shutil
import traceback
import logging
from datetime import datetime
import time
import tempfile
try:
    DEBUG_DIR = os.path.join(os.path.dirname(__file__), '.debug')
    os.makedirs(DEBUG_DIR, exist_ok=True)
except Exception:
    DEBUG_DIR = os.path.join(tempfile.gettempdir(), '.debug')
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
    except Exception:
        pass


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
    # 直接固定為 \input\SolarRecord(260228)_d_forWh_WithCodis.csv
    candidate = os.path.normpath(os.path.join(base, 'input', 'SolarRecord(260228)_d_forWh_WithCodis.csv'))
    if os.path.exists(candidate):
        return candidate

    # fallback: 仍保留原來的候選清單
    candidates = [
        os.path.join(base, '..', 'csv', 'SolarRecord_260310_1829-daily-1d.csv'),
        os.path.join(base, '..', 'csv', 'SolarRecord(260228)_d_forWh_WithCodis.csv'),
        os.path.join(base, 'csv', '2000--202602-d-forWh_4b.csv'),
        os.path.join(base, '..', 'csv', '2000--202602-d-forWh_4b.csv'),
        # os.path.join(base, '..', 'csv', '2000--202602-d-forWh.csv'),
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


def get_output_parent(base: str, timestamp: str | None = None, start_count: int = 1, name_hint: str | None = None, user_tag: str | None = None):
    """Create a unique output parent directory.

    If `timestamp` is provided it will be used; otherwise current time is used.
    `start_count` allows callers to request a preferred starting count (it will
    still increment until an unused name is found).
    Returns (output_parent, timestamp, count_used).
    """
    script_name = name_hint or os.path.splitext(os.path.basename(__file__))[0]
    if timestamp is None:
        timestamp = pd.Timestamp.now().strftime('%y%m%d_%H%M%S')
    count = int(start_count) if start_count is not None else 1
    # create nested folder structure under base: {base}/{script_name}/{user_tag?}/{timestamp}
    while True:
        if count == 1:
            if user_tag:
                candidate = os.path.join(base, script_name, user_tag, timestamp)
            else:
                candidate = os.path.join(base, script_name, timestamp)
        else:
            if user_tag:
                candidate = os.path.join(base, script_name, user_tag, f'{timestamp}_{count}')
            else:
                candidate = os.path.join(base, script_name, f'{timestamp}_{count}')
        if not os.path.exists(candidate):
            output_parent = candidate
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


# Strict override protection flags (預設保護開啟)
FORBID_MODEL_OVERRIDE = True
ALLOW_TRANSFORMER_RETRY = True
ON_OVERRIDE_ACTION = 'warn_and_skip'
ENABLE_NUM_VALIDATIONS_BACKOFF = True
NUM_VALIDATIONS_BACKOFF_SEQUENCE = ['auto', 2, 1, 0]


def _parse_bool(s):
    if s is None:
        return None
    if isinstance(s, bool):
        return s
    low = str(s).lower()
    if low in ('1', 'true', 'yes', 'y'):
        return True
    if low in ('0', 'false', 'no', 'n'):
        return False
    return None


def _normalize_model_names(model_list):
    if model_list is None:
        return []
    if isinstance(model_list, str):
        return [model_list]
    names = []
    try:
        for m in model_list:
            if isinstance(m, dict):
                name = m.get('model') or m.get('Model')
                if name:
                    names.append(str(name))
            else:
                names.append(str(m))
    except Exception:
        return [str(model_list)]
    return names


def _parse_num_validations_backoff_sequence(raw_sequence):
    seq = []
    for token in str(raw_sequence).split(','):
        t = str(token).strip()
        if not t:
            continue
        if t.lower() == 'auto':
            seq.append('auto')
            continue
        try:
            iv = int(t)
        except Exception as exc:
            raise ValueError(f"invalid backoff token: {t}") from exc
        if iv < 0:
            raise ValueError(f"backoff value must be >= 0: {iv}")
        seq.append(iv)
    # de-duplicate while preserving order
    dedup = []
    seen = set()
    for x in seq:
        key = str(x).lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(x)
    if not dedup:
        raise ValueError('num_validations_backoff_sequence is empty')
    return dedup


def _is_num_validations_backoff_retriable_error(message: str):
    m = str(message or '').lower()
    patterns = [
        'forecast_length is too large for training data',
        'num_validations/num_indices too high',
        'too many training validations for length of data provided',
    ]
    return any(p in m for p in patterns)


def validate_model_data_compatibility(model_list, series_columns, horizon=None):
    model_names = _normalize_model_names(model_list)
    upper_names = set([m.upper() for m in model_names])
    if 'VAR' in upper_names and int(series_columns) < 2:
        hz = f', horizon={horizon}' if horizon is not None else ''
        raise ValueError(
            f"Invalid model/data combination{hz}: model_list={model_names} includes VAR, "
            f"but series_columns={series_columns}. VAR requires at least 2 variables."
        )


def handle_model_override(original, attempted, out_root=None, horizon=None, action='warn_and_skip'):
    try:
        if isinstance(original, str):
            orig_list = [original]
        elif original is None:
            orig_list = []
        else:
            orig_list = list(original)
    except Exception:
        orig_list = []
    try:
        if isinstance(attempted, str):
            att_list = [attempted]
        elif attempted is None:
            att_list = []
        else:
            att_list = list(attempted)
    except Exception:
        att_list = []

    orig_set = set([str(x) for x in orig_list])
    att_set = set([str(x) for x in att_list])
    if att_set.issubset(orig_set):
        return True
    msg = f"Model override detected: original={orig_list}, attempted={att_list}, horizon={horizon}"
    print(msg)
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        logp = os.path.join(DEBUG_DIR, 'model_override.log')
        with open(logp, 'a', encoding='utf-8') as lf:
            lf.write(datetime.now().isoformat() + ' - ' + msg + '\n')
    except Exception:
        pass
    if action == 'fail':
        raise RuntimeError(msg)
    return False


def save_effective_settings(output_dir, inp, horizons, timestamp, ats_kwargs_template):
    settings = {
        'input_file': inp,
        'horizons': horizons,
        'ats_kwargs_template': ats_kwargs_template,
        'python_version': platform.python_version(),
        'output_dir': output_dir,
        'timestamp': timestamp,
        'FORBID_MODEL_OVERRIDE': FORBID_MODEL_OVERRIDE,
        'ALLOW_TRANSFORMER_RETRY': ALLOW_TRANSFORMER_RETRY,
        'ON_OVERRIDE_ACTION': ON_OVERRIDE_ACTION,
    }
    try:
        settings['random_seed'] = globals().get('random_seed')
    except Exception:
        settings['random_seed'] = None
    ef_path = os.path.join(output_dir, 'effective_settings.json')
    with open(ef_path, 'w', encoding='utf-8') as ef:
        json.dump(settings, ef, ensure_ascii=False, indent=2)
    print('Saved effective settings to', ef_path)


def fallback_prophet_predict(dfp_train_local, horizon_local, ProphetClass):
    """Use ProphetClass to fit on dfp_train_local and return last `horizon_local` rows
    as a DataFrame indexed by `ds` with a column renamed to `Wh` (from yhat).
    This is extracted for unit testing and reusability.
    """
    dfp = dfp_train_local[['ds', 'y']].dropna()
    m = ProphetClass()
    m.fit(dfp)
    future = m.make_future_dataframe(periods=horizon_local, freq='D')
    fc = m.predict(future)
    fc_tail = fc[['ds', 'yhat']].set_index('ds').tail(horizon_local)
    fc_tail = fc_tail.rename(columns={'yhat': 'Wh'})
    return fc_tail


def _to_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    if isinstance(value, pd.DataFrame):
        try:
            return [{k: _to_jsonable(v) for k, v in row.items()} for row in value.to_dict(orient='records')]
        except Exception:
            return str(value)
    if isinstance(value, pd.Series):
        try:
            return {str(k): _to_jsonable(v) for k, v in value.to_dict().items()}
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    try:
        return str(value)
    except Exception:
        return None


def _extract_best_model_info(model):
    info = {}
    keys = [
        'best_model_name',
        'best_model',
        'best_model_params',
        'best_model_transformation_params',
        'best_model_transformations',
    ]
    for key in keys:
        try:
            info[key] = _to_jsonable(getattr(model, key, None))
        except Exception:
            info[key] = None

    # Best-effort normalization for display/import hints.
    if not info.get('best_model_name'):
        bm = info.get('best_model')
        try:
            if isinstance(bm, list) and bm and isinstance(bm[0], dict):
                info['best_model_name'] = bm[0].get('Model') or bm[0].get('model')
            elif isinstance(bm, dict):
                info['best_model_name'] = bm.get('Model') or bm.get('model')
        except Exception:
            pass
    return info


def _derive_best_template_df(model):
    # Primary source: model.best_model (often already template-like DataFrame).
    try:
        bm = getattr(model, 'best_model', None)
        if isinstance(bm, pd.DataFrame) and not bm.empty:
            return bm.copy()
        if isinstance(bm, dict):
            return pd.DataFrame([bm])
        if isinstance(bm, list) and bm and isinstance(bm[0], dict):
            return pd.DataFrame(bm)
    except Exception:
        pass

    # Secondary source: model.best_model_params + transformation params.
    try:
        name = getattr(model, 'best_model_name', None)
        mp = getattr(model, 'best_model_params', None)
        tp = getattr(model, 'best_model_transformation_params', None)
        if name is not None or mp is not None or tp is not None:
            row = {
                'Model': name,
                'ModelParameters': mp if isinstance(mp, str) else json.dumps(_to_jsonable(mp), ensure_ascii=False),
                'TransformationParameters': tp if isinstance(tp, str) else json.dumps(_to_jsonable(tp), ensure_ascii=False),
                'Ensemble': 0,
            }
            return pd.DataFrame([row])
    except Exception:
        pass

    return None


def save_horizon_best_artifacts(model, horizon_dir, horizon):
    result = {
        'best_model_info_path': None,
        'best_template_csv_path': None,
        'best_template_json_path': None,
        'best_model_name': None,
    }
    os.makedirs(horizon_dir, exist_ok=True)

    best_info = _extract_best_model_info(model)
    result['best_model_name'] = best_info.get('best_model_name')

    best_info_path = os.path.join(horizon_dir, f'best_model_info_{horizon}d.json')
    try:
        with open(best_info_path, 'w', encoding='utf-8') as bf:
            json.dump(best_info, bf, ensure_ascii=False, indent=2)
        result['best_model_info_path'] = best_info_path
        print('Saved best model info to', best_info_path)
    except Exception as e:
        print('Failed to save best model info:', e)

    template_df = _derive_best_template_df(model)
    if template_df is None or template_df.empty:
        return result

    template_csv_path = os.path.join(horizon_dir, f'best_template_{horizon}d.csv')
    template_json_path = os.path.join(horizon_dir, f'best_template_{horizon}d.json')

    try:
        template_df.to_csv(template_csv_path, index=False)
        result['best_template_csv_path'] = template_csv_path
        print('Saved best template CSV to', template_csv_path)
    except Exception as e:
        print('Failed to save best template CSV:', e)

    try:
        payload = {
            'columns': [str(c) for c in template_df.columns.tolist()],
            'records': _to_jsonable(template_df),
        }
        with open(template_json_path, 'w', encoding='utf-8') as tf:
            json.dump(payload, tf, ensure_ascii=False, indent=2)
        result['best_template_json_path'] = template_json_path
        print('Saved best template JSON to', template_json_path)
    except Exception as e:
        print('Failed to save best template JSON:', e)

    return result


def save_iteration_best_artifacts(output_dir, timestamp, iteration, horizon_results):
    if not isinstance(horizon_results, list):
        horizon_results = []

    valid_results = []
    for r in horizon_results:
        if not isinstance(r, dict):
            continue
        if r.get('status') != 'success':
            continue
        valid_results.append(r)

    if not valid_results:
        return None

    mase_candidates = []
    for r in valid_results:
        v = r.get('mase_lag1')
        if isinstance(v, (int, float)):
            mase_candidates.append(r)
    mase_best = min(mase_candidates, key=lambda x: float(x.get('mase_lag1'))) if mase_candidates else None

    summary = {
        'timestamp': timestamp,
        'iteration': int(iteration),
        'mase_lag1_best': {
            'horizon': mase_best.get('horizon') if mase_best else None,
            'mase_lag1': mase_best.get('mase_lag1') if mase_best else None,
            'best_model_name': mase_best.get('best_model_name') if mase_best else None,
            'artifacts': mase_best.get('artifacts') if mase_best else None,
        },
        'autots_best_by_horizon': [
            {
                'horizon': r.get('horizon'),
                'best_model_name': r.get('best_model_name'),
                'mase_lag1': r.get('mase_lag1'),
                'artifacts': r.get('artifacts'),
            }
            for r in valid_results
        ],
    }

    summary_path = os.path.join(output_dir, 'iteration_best_summary.json')
    try:
        with open(summary_path, 'w', encoding='utf-8') as sf:
            json.dump(_to_jsonable(summary), sf, ensure_ascii=False, indent=2)
        print('Saved iteration best summary to', summary_path)
    except Exception as e:
        print('Failed to save iteration best summary:', e)

    if mase_best:
        try:
            src_csv = ((mase_best.get('artifacts') or {}).get('best_template_csv_path'))
            src_json = ((mase_best.get('artifacts') or {}).get('best_template_json_path'))
            if src_csv and os.path.exists(src_csv):
                dst_csv = os.path.join(output_dir, 'iteration_best_template.csv')
                shutil.copyfile(src_csv, dst_csv)
                print('Saved iteration best template CSV to', dst_csv)
            if src_json and os.path.exists(src_json):
                dst_json = os.path.join(output_dir, 'iteration_best_template.json')
                shutil.copyfile(src_json, dst_json)
                print('Saved iteration best template JSON to', dst_json)
        except Exception as e:
            print('Failed to save iteration best template artifacts:', e)

    return summary_path


def run_horizon(AutoTS, ser, horizon, output_dir, timestamp, count, n_jobs, max_generations,
                num_validations, transformer_list, model_list, ensemble, metric_weighting=None,
                enable_num_validations_backoff=True, num_validations_backoff_sequence=None):
    # `output_dir`, `timestamp`, `count` are determined by caller (before horizons loop)
    start_perf = time.perf_counter()
    start_iso = datetime.now().isoformat()
    nv_attempts = []
    horizon_result = {
        'horizon': int(horizon),
        'status': 'failed',
        'reason': None,
        'mase_lag1': None,
        'best_model_name': None,
        'artifacts': {},
    }

    def _run():
        # use a local copy to avoid assigning to outer-scope name
        local_model_list = model_list
        validate_model_data_compatibility(local_model_list, ser.shape[1], horizon=horizon)

        if len(ser) <= horizon:
            print(f'資料筆數太少無法切出測試集 (len({len(ser)}) <= horizon {horizon})，跳過')
            horizon_result['reason'] = 'insufficient_data_for_horizon'
            return horizon_result

        train_ser = ser.iloc[:-horizon].copy()
        test_ser = ser.iloc[-horizon:].copy()

        dfp_train = train_ser.reset_index()
        dfp_train.columns = ['ds', 'y']
        dfp_train['ds'] = pd.to_datetime(dfp_train['ds'], errors='coerce')
        dfp_train = dfp_train.dropna(subset=['ds'])

        # Defensive: ensure DataFrame is a writable copy and numeric where required.
        dfp_train = dfp_train.copy()
        dfp_train['y'] = pd.to_numeric(dfp_train['y'], errors='coerce')
        dfp_train = dfp_train.dropna(subset=['y'])
        dfp_train['y'] = dfp_train['y'].astype(float)

        # Accept AutoTS native 'auto' or pass through explicit launcher/user value unchanged.
        if isinstance(num_validations, str) and num_validations.strip().lower() == 'auto':
            base_num_validations = 'auto'
        else:
            base_num_validations = max(1, int(num_validations))

        # Build attempt list: base value first, then optional backoff sequence.
        attempt_num_validations = [base_num_validations]
        if enable_num_validations_backoff:
            seq = num_validations_backoff_sequence or ['auto', 2, 1, 0]
            for x in seq:
                key = str(x).lower()
                if all(str(a).lower() != key for a in attempt_num_validations):
                    attempt_num_validations.append(x)

        print(
            f"Using num_validations attempts={attempt_num_validations} "
            f"for horizon={horizon} (backoff={enable_num_validations_backoff})"
        )

        # 如果 model_list 裡有 dict（CLI 以 JSON 傳入的 model 參數），把它轉成 AutoTS 可接受的 initial_template
        horizon_dir = os.path.join(output_dir, str(horizon))
        os.makedirs(horizon_dir, exist_ok=True)
        forecast_df = None
        for attempt_index, attempt_nv in enumerate(attempt_num_validations, start=1):
            # Per-attempt value: allow 'auto' or integer >= 0.
            if isinstance(attempt_nv, str) and attempt_nv.strip().lower() == 'auto':
                autots_num_validations = 'auto'
            else:
                autots_num_validations = max(0, int(attempt_nv))

            print(
                f'Attempt {attempt_index}/{len(attempt_num_validations)} '
                f'for horizon={horizon} with num_validations={autots_num_validations}'
            )

            current_model_list = local_model_list
            ats_kwargs = dict(
                model_list=current_model_list,
                forecast_length=horizon,
                frequency='D',
                transformer_list=transformer_list,
                n_jobs=n_jobs,
                max_generations=max_generations,
                num_validations=autots_num_validations,
                validation_method='backwards',
                ensemble=ensemble,
                prediction_interval=0.9,
            )
            try:
                import pandas as _pd

                if isinstance(current_model_list, list) and any(isinstance(m, dict) for m in current_model_list):
                    rows = []
                    model_names = []
                    for m in current_model_list:
                        if isinstance(m, dict):
                            model_name = m.get('model') or m.get('Model')
                            params = m.get('model_params') or m.get('ModelParameters') or {}
                            mp = json.dumps(params) if not isinstance(params, str) else params
                            tp = json.dumps({"fillna": "zero", "transformations": {}, "transformation_params": {}})
                            rows.append({'Model': model_name, 'ModelParameters': mp, 'TransformationParameters': tp, 'Ensemble': 0})
                            model_names.append(model_name)
                    if rows:
                        ats_kwargs['initial_template'] = _pd.DataFrame(rows)
                        ats_kwargs['model_list'] = list(model_names)
                        current_model_list = list(model_names)
            except Exception:
                pass

            if metric_weighting:
                try:
                    ats_kwargs['metric_weighting'] = metric_weighting
                except Exception:
                    pass

            print(f'Instantiating AutoTS with model_list={current_model_list} and horizon={horizon}...')
            try:
                model = AutoTS(**{k: v for k, v in ats_kwargs.items() if k != 'forecast_length'})
            except ValueError as e:
                msg = str(e)
                if 'transformer_list' in msg or 'alias not recognized' in msg:
                    print('AutoTS: transformer_list caused error')
                    if not ALLOW_TRANSFORMER_RETRY:
                        print('Transformer retry disabled by ALLOW_TRANSFORMER_RETRY; re-raising')
                        raise
                    print('Retrying with AutoTS default transformers')
                    temp_kwargs = dict(ats_kwargs)
                    if 'transformer_list' in temp_kwargs:
                        temp_kwargs.pop('transformer_list', None)
                    try:
                        model = AutoTS(**{k: v for k, v in temp_kwargs.items() if k != 'forecast_length'})
                    except Exception:
                        print('AutoTS: fallback to default model_list and transformers')
                        if FORBID_MODEL_OVERRIDE:
                            allow = handle_model_override(current_model_list, 'default', out_root=horizon_dir if 'horizon_dir' in locals() else None, horizon=horizon, action=ON_OVERRIDE_ACTION)
                            if not allow:
                                print('Fallback to default model_list blocked by FORBID_MODEL_OVERRIDE; skipping this horizon')
                                horizon_result['reason'] = 'model_override_blocked'
                                return horizon_result
                        model = AutoTS(model_list='default')
                else:
                    raise

            try:
                model.forecast_length = horizon
            except Exception:
                pass

            try:
                print(f'Fitting AutoTS (Prophet only) on training set (horizon={horizon})...')
                model = model.fit(dfp_train.copy(deep=True), date_col='ds', value_col='y')
                print('Predicting', horizon, 'steps ahead...')
                pred = model.predict(forecast_length=horizon)
                if hasattr(pred, 'forecast'):
                    forecast_df = pred.forecast
                elif isinstance(pred, pd.DataFrame):
                    forecast_df = pred
                else:
                    forecast_df = pd.DataFrame(pred)
                nv_attempts.append({
                    'attempt_index': attempt_index,
                    'num_validations': autots_num_validations,
                    'status': 'success',
                    'error': None,
                })
                break
            except Exception as e:
                err_text = str(e)
                retriable = bool(enable_num_validations_backoff and _is_num_validations_backoff_retriable_error(err_text))
                nv_attempts.append({
                    'attempt_index': attempt_index,
                    'num_validations': autots_num_validations,
                    'status': 'failed',
                    'error': err_text,
                    'retriable': retriable,
                })
                print('AutoTS fit/predict failed:', err_text)
                err_msg = (
                    f'AutoTS fit/predict failed for horizon {horizon} '
                    f'(attempt {attempt_index}, num_validations={autots_num_validations}): {e}'
                )
                try:
                    logging.error(err_msg, exc_info=True)
                except Exception:
                    pass
                try:
                    os.makedirs(DEBUG_DIR, exist_ok=True)
                    with open(os.path.join(DEBUG_DIR, 'autots_fail.log'), 'a', encoding='utf-8') as lf:
                        lf.write(datetime.now().isoformat() + ' - ' + err_msg + '\n')
                        lf.write(traceback.format_exc() + '\n')
                except Exception:
                    pass
                if retriable and attempt_index < len(attempt_num_validations):
                    next_nv = attempt_num_validations[attempt_index]
                    print(
                        f'Retriable validation error detected; retrying with '
                        f'num_validations={next_nv}'
                    )
                    continue
                print('Skipping this horizon due to AutoTS failure.')
                horizon_result['reason'] = 'autots_fit_predict_failed'
                return horizon_result

        if forecast_df is None:
            print('Skipping this horizon due to AutoTS failure.')
            horizon_result['reason'] = 'autots_forecast_not_available'
            return horizon_result

        horizon_dir = os.path.join(output_dir, str(horizon))
        os.makedirs(horizon_dir, exist_ok=True)

        # Save AutoTS best model/template artifacts for future import/audit.
        try:
            best_artifacts = save_horizon_best_artifacts(model, horizon_dir, horizon)
            horizon_result['artifacts'] = best_artifacts
            horizon_result['best_model_name'] = best_artifacts.get('best_model_name')
        except Exception as e:
            print('Failed to save horizon best artifacts:', e)

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
                    horizon_result['mase_lag1'] = scores.get('MASE_lag1')
                except Exception:
                    horizon_result['mase_lag1'] = None

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

        horizon_result['status'] = 'success'
        return horizon_result

    rv = None
    try:
        rv = _run()
        return rv
    finally:
        try:
            end_perf = time.perf_counter()
            end_iso = datetime.now().isoformat()
            duration_s = end_perf - start_perf
            exc_type, exc_value, exc_tb = sys.exc_info()
            exc_text = None
            if exc_type is not None and exc_value is not None:
                try:
                    exc_text = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
                except Exception:
                    exc_text = str(exc_value)
            runtime_info = {
                'horizon': horizon,
                'timestamp': timestamp,
                'count': count,
                'start_iso': start_iso,
                'end_iso': end_iso,
                'duration_s': float(duration_s),
                'random_seed': globals().get('random_seed'),
                'num_validations_backoff_enabled': bool(enable_num_validations_backoff),
                'num_validations_attempts': nv_attempts,
                'exception': exc_text,
            }
            # try writing into horizon_dir, then output_dir, then debug dir, then temp dir
            write_paths = [os.path.join(output_dir, str(horizon)), output_dir, DEBUG_DIR, tempfile.gettempdir()]
            written = False
            for p in write_paths:
                try:
                    os.makedirs(p, exist_ok=True)
                    outp = os.path.join(p, f'horizon_runtime_{timestamp}_{count}.json')
                    with open(outp, 'w', encoding='utf-8') as rf:
                        json.dump(runtime_info, rf, ensure_ascii=False, indent=2)
                    print('Saved horizon runtime to', outp)
                    written = True
                    break
                except Exception:
                    continue
            if not written:
                print('Failed to save horizon runtime for horizon', horizon)
        except Exception:
            pass


def main():
    import argparse

    parser = argparse.ArgumentParser(description='AutoTS Prophet multi-horizon runner')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug logging and exception capture')
    parser.add_argument('--debug_log', type=str, default=None, help='Path to debug log file (optional)')
    parser.add_argument('--horizons', nargs='+', type=int, default=[3, 6, 9],
                        help='List of horizons to run, e.g. --horizons 3 6 9')
    parser.add_argument('--n_jobs', type=int, default=-1, help='AutoTS n_jobs')
    parser.add_argument('--max_generations', type=int, default=1, help='AutoTS max_generations')
    parser.add_argument('--num_validations', type=str, default='auto', help="AutoTS num_validations. Use 'auto' or a positive integer")
    parser.add_argument('--enable_num_validations_backoff', type=str, default='True',
                        help='Enable horizon-level num_validations backoff. true/false')
    parser.add_argument('--num_validations_backoff_sequence', type=str, default='auto,2,1,0',
                        help="Fallback sequence for num_validations when enabled. Example: 'auto,2,1,0'")
    parser.add_argument('--transformer_list', nargs='+', default=['default'], help='AutoTS transformer_list')
    parser.add_argument('--model_list', nargs='+', default=None, help='AutoTS model_list')
    parser.add_argument('--ensemble', default=None, help='AutoTS ensemble parameter (None, auto, simple, etc.)')
    parser.add_argument('--input_file', type=str, default=None, help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, default=None, help='Override output directory (path). If relative, resolved against script directory')
    parser.add_argument('--output_tag', type=str, default=None, help='Optional tag included in output path (e.g. experiment name)')
    parser.add_argument('--metric_weighting', type=str, default=None, help='JSON string or path to JSON file for AutoTS metric_weighting.')
    parser.add_argument('--loop', action='store_true', help='Run horizons repeatedly until interrupted')
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Random seed (optional). If not provided, a random 32-bit seed will be generated.')
    parser.add_argument('--forbid_model_override')
    parser.add_argument('--allow_transformer_retry')
    parser.add_argument('--on_override_action')
    args = parser.parse_args()

    def _normalize_num_validations(raw_value):
        s = str(raw_value).strip()
        if s.lower() == 'auto':
            return 'auto'
        try:
            iv = int(s)
        except Exception as exc:
            raise ValueError("--num_validations must be 'auto' or a positive integer") from exc
        if iv < 1:
            raise ValueError("--num_validations must be 'auto' or a positive integer")
        return iv

    try:
        args.num_validations = _normalize_num_validations(args.num_validations)
    except ValueError as e:
        parser.error(str(e))

    parsed_backoff = _parse_bool(args.enable_num_validations_backoff)
    if parsed_backoff is None:
        parser.error("--enable_num_validations_backoff must be true or false")
    args.enable_num_validations_backoff = parsed_backoff

    try:
        args.num_validations_backoff_sequence = _parse_num_validations_backoff_sequence(args.num_validations_backoff_sequence)
    except ValueError as e:
        parser.error(f"--num_validations_backoff_sequence invalid: {e}")

    # Debug logging / exception capture
    if getattr(args, 'debug', False):
        debug_log_path = args.debug_log or os.path.join(DEBUG_DIR, 'child_debug.log')
        try:
            os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)
        except Exception:
            pass
        try:
            logging.basicConfig(filename=debug_log_path, level=logging.DEBUG,
                                format='%(asctime)s %(levelname)s %(message)s')
        except Exception:
            pass

        def _child_excepthook(exc_type, exc_value, exc_tb):
            tb = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
            try:
                logging.error('Uncaught exception in child:\n%s', tb)
            except Exception:
                pass
            try:
                logp = os.path.join(DEBUG_DIR, 'launcher_errors.log')
                with open(logp, 'a', encoding='utf-8') as lf:
                    lf.write(datetime.now().isoformat() + ' - child exception: ' + tb + '\n')
            except Exception:
                pass
            try:
                sys.stderr.write(tb)
            except Exception:
                pass

        sys.excepthook = _child_excepthook

    # 支援從 CLI 傳入 JSON 字串作為 model_list 的元素，會嘗試解析成 dict
    try:
        import json as _json

        parsed_models = []
        for m in args.model_list:
            if isinstance(m, str):
                try:
                    parsed_models.append(_json.loads(m))
                except Exception:
                    parsed_models.append(m)
            else:
                parsed_models.append(m)
        args.model_list = parsed_models
    except Exception:
        pass

    # Parse and normalise metric_weighting argument (accept JSON, file, or simple k:v list)
    def _parse_metric_weighting_arg(arg):
        if not arg:
            return None
        try:
            if os.path.exists(arg):
                with open(arg, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return json.loads(arg)
        except Exception:
            d = {}
            for p in str(arg).split(','):
                if not p.strip():
                    continue
                if ':' in p:
                    k, v = p.split(':', 1)
                elif '=' in p:
                    k, v = p.split('=', 1)
                else:
                    continue
                try:
                    d[k.strip()] = float(v.strip())
                except Exception:
                    d[k.strip()] = v.strip()
            return d

    def _normalize_metric_weighting(mw):
        if not isinstance(mw, dict):
            return mw
        def _to_number(v):
            try:
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return float(v)
                if isinstance(v, str):
                    s = v.strip().strip('"\'""').strip('{} ').strip()
                    return float(s)
            except Exception:
                pass
            return v

        out = {}
        for k, v in mw.items():
            kk = str(k)
            vv = _to_number(v)
            if kk.lower().endswith('_weighting'):
                out[kk.lower()] = vv
                continue
            u = kk.upper()
            if u == 'MAE':
                out['mae_weighting'] = vv
            elif u == 'SMAPE':
                out['smape_weighting'] = vv
            elif u == 'RMSE':
                out['rmse_weighting'] = vv
            elif u == 'MASE':
                out['mase_weighting'] = vv
            else:
                out[kk.lower()] = vv

        try:
            others = [float(x) for kk, x in out.items() if kk != 'mae_weighting' and isinstance(x, (int, float))]
            max_other = max(others) if others else 0.0
        except Exception:
            max_other = 0.0
        try:
            out['mae_weighting'] = max(float(out.get('mae_weighting', 0)), max_other + 1)
        except Exception:
            out['mae_weighting'] = max_other + 1
        return out

    metric_weighting = _parse_metric_weighting_arg(getattr(args, 'metric_weighting', None))
    metric_weighting = _normalize_metric_weighting(metric_weighting)

    base = os.path.dirname(__file__)
    # 將預設輸出根目錄移到 script 的 output 子目錄
    output_root = os.path.join(base, 'output')
    os.makedirs(output_root, exist_ok=True)
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
    num_validations = args.num_validations
    transformer_list = args.transformer_list
    model_list = args.model_list

    # Normalize launcher/CLI inputs: treat ['default'] as the 'default' sentinel
    try:
        if isinstance(model_list, list) and len(model_list) == 1 and model_list[0] == 'default':
            model_list = 'default'
    except Exception:
        pass
    try:
        if isinstance(transformer_list, list) and len(transformer_list) == 1 and transformer_list[0] == 'default':
            transformer_list = 'default'
    except Exception:
        pass

    ensemble = None if args.ensemble in (None, 'None', 'none') else args.ensemble

    # CLI flag overrides for strict override behavior
    if getattr(args, 'forbid_model_override', None) is not None:
        b = _parse_bool(args.forbid_model_override)
        if b is not None:
            globals()['FORBID_MODEL_OVERRIDE'] = b
    if getattr(args, 'allow_transformer_retry', None) is not None:
        b = _parse_bool(args.allow_transformer_retry)
        if b is not None:
            globals()['ALLOW_TRANSFORMER_RETRY'] = b
    if getattr(args, 'on_override_action', None) is not None:
        globals()['ON_OVERRIDE_ACTION'] = args.on_override_action

    globals()['ENABLE_NUM_VALIDATIONS_BACKOFF'] = bool(args.enable_num_validations_backoff)
    globals()['NUM_VALIDATIONS_BACKOFF_SEQUENCE'] = list(args.num_validations_backoff_sequence)

    # Enforce repository policy: always forbid AutoTS model override regardless of CLI
    try:
        globals()['FORBID_MODEL_OVERRIDE'] = True
    except Exception:
        pass

    validate_model_data_compatibility(model_list, ser.shape[1])
    print(
        'Effective config:',
        f'model_list={model_list},',
        f'transformer_list={transformer_list},',
        f'ensemble={ensemble},',
        f'num_validations={num_validations},',
        f'enable_num_validations_backoff={args.enable_num_validations_backoff},',
        f'num_validations_backoff_sequence={args.num_validations_backoff_sequence},',
        f'FORBID_MODEL_OVERRIDE={FORBID_MODEL_OVERRIDE},',
        f'ALLOW_TRANSFORMER_RETRY={ALLOW_TRANSFORMER_RETRY},',
        f'ON_OVERRIDE_ACTION={ON_OVERRIDE_ACTION}'
    )

    AutoTS = init_autots()

    # ------- Random seed setup -------
    random_seed = getattr(args, 'random_seed', None)
    if random_seed is None:
        try:
            import random as _tmp_rand
            random_seed = _tmp_rand.randint(0, 2**31 - 1)
        except Exception:
            random_seed = 0
    try:
        globals()['random_seed'] = int(random_seed)
    except Exception:
        pass

    try:
        import random as _py_random
        _py_random.seed(random_seed)
    except Exception:
        pass
    try:
        np.random.seed(random_seed)
    except Exception:
        pass
    try:
        os.environ['PYTHONHASHSEED'] = str(random_seed)
    except Exception:
        pass

    print('Using random_seed:', random_seed)
    # ------- end seed setup -------

    # sanitize helper and derive safe_model_name for output folder naming
    def _sanitize_name(name):
        s = str(name or '').strip()
        s = re.sub(r'[^A-Za-z0-9._-]+', '_', s)
        if not s:
            s = os.path.splitext(os.path.basename(__file__))[0]
        reserved = {'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'LPT1', 'LPT2', 'LPT3'}
        if s.upper() in reserved:
            s = '_' + s
        return s

    try:
        raw_names = []
        if isinstance(model_list, list):
            for m in model_list:
                if isinstance(m, dict):
                    mn = m.get('model') or m.get('Model')
                    if mn:
                        raw_names.append(str(mn))
                    else:
                        raw_names.append(json.dumps(m.get('model_params') or m.get('ModelParameters') or {}))
                else:
                    raw_names.append(str(m))
        else:
            raw_names.append(str(model_list))
        if len(raw_names) == 1:
            raw_model_name = raw_names[0]
        else:
            raw_model_name = '+'.join([n for n in raw_names if n])
        safe_model_name = _sanitize_name(raw_model_name)
    except Exception:
        safe_model_name = _sanitize_name(None)

    # sanitize user-provided output tag (if any)
    if getattr(args, 'output_tag', None):
        safe_output_tag = _sanitize_name(args.output_tag)
    else:
        safe_output_tag = None

    iteration = 0
    while True:
        iteration += 1
        print(f'Iteration {iteration} start...')
        iteration_horizon_results = []
        # 決定 timestamp 與 count（只做一次，置於父目錄名稱中）
        if getattr(args, 'output_dir', None):
            # Treat provided --output_dir as a root; child will create
            # {root}/{model}/{tag?}/{timestamp}
            output_root_override = os.path.normpath(args.output_dir)
            if not os.path.isabs(output_root_override):
                output_root_override = os.path.normpath(os.path.join(output_root, output_root_override))
            output_dir, timestamp, count = get_output_parent(output_root_override, name_hint=safe_model_name, user_tag=safe_output_tag)
        else:
            # 預設輸出目錄改為 script/output/{model}/{tag?}/{timestamp}
            output_dir, timestamp, count = get_output_parent(output_root, name_hint=safe_model_name, user_tag=safe_output_tag)
        iter_start_perf = time.perf_counter()
        iter_start_iso = datetime.now().isoformat()
        try:
            for horizon in horizons:
                horizon_result = run_horizon(AutoTS, ser, horizon, output_dir, timestamp, count,
                                             n_jobs=n_jobs,
                                             max_generations=max_generations,
                                             num_validations=num_validations,
                                             enable_num_validations_backoff=args.enable_num_validations_backoff,
                                             num_validations_backoff_sequence=args.num_validations_backoff_sequence,
                                             transformer_list=transformer_list,
                                             model_list=model_list,
                                             ensemble=ensemble,
                                             metric_weighting=metric_weighting)
                iteration_horizon_results.append(horizon_result)
                # 每跑完一個 horizon，增加 count，用於下一個 horizon 的檔名 suffix
                count += 1
        finally:
            try:
                iter_end_perf = time.perf_counter()
                iter_end_iso = datetime.now().isoformat()
                iter_duration_s = iter_end_perf - iter_start_perf
                iter_info = {
                    'iteration': iteration,
                    'start_iso': iter_start_iso,
                    'end_iso': iter_end_iso,
                    'duration_s': float(iter_duration_s),
                    'horizons': horizons,
                    'timestamp': timestamp,
                    'count_end': count,
                    'random_seed': globals().get('random_seed'),
                }
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    iter_path = os.path.join(output_dir, f'iteration_runtime_{timestamp}_{iteration}.json')
                    with open(iter_path, 'w', encoding='utf-8') as itf:
                        json.dump(iter_info, itf, ensure_ascii=False, indent=2)
                    print('Saved iteration runtime to', iter_path)
                except Exception:
                    try:
                        fallback = tempfile.gettempdir()
                        iter_path = os.path.join(fallback, f'iteration_runtime_{timestamp}_{iteration}.json')
                        with open(iter_path, 'w', encoding='utf-8') as itf:
                            json.dump(iter_info, itf, ensure_ascii=False, indent=2)
                        print('Saved iteration runtime to', iter_path)
                    except Exception:
                        print('Failed to save iteration runtime for iteration', iteration)
            except Exception:
                pass

        # 將 effective settings 儲存在本次迭代的 output 目錄
        try:
            ats_kwargs_template = {
                'model_list': model_list,
                'frequency': 'D',
                'transformer_list': transformer_list,
                'n_jobs': int(n_jobs),
                'max_generations': int(max_generations),
                'num_validations': num_validations if isinstance(num_validations, str) else int(num_validations),
                'enable_num_validations_backoff': bool(args.enable_num_validations_backoff),
                'num_validations_backoff_sequence': list(args.num_validations_backoff_sequence),
                'validation_method': 'backwards',
                'ensemble': ensemble,
                'prediction_interval': 0.9,
                'metric_weighting': metric_weighting,
            }
            save_effective_settings(output_dir, inp, horizons, timestamp, ats_kwargs_template)
        except Exception:
            pass

        # Save iteration-level best summary/template (MASE best + AutoTS best by horizon).
        try:
            save_iteration_best_artifacts(output_dir, timestamp, iteration, iteration_horizon_results)
        except Exception as e:
            print('Failed to save iteration best artifacts:', e)

        if not args.loop:
            break


if __name__ == '__main__':
    script_start_perf = time.perf_counter()
    script_start_iso = datetime.now().isoformat()
    try:
        main()
    finally:
        try:
            script_end_perf = time.perf_counter()
            script_end_iso = datetime.now().isoformat()
            script_duration_s = script_end_perf - script_start_perf
            script_info = {
                'start_iso': script_start_iso,
                'end_iso': script_end_iso,
                'duration_s': float(script_duration_s),
                'random_seed': globals().get('random_seed'),
            }
            try:
                out_dir = DEBUG_DIR
                os.makedirs(out_dir, exist_ok=True)
                out_name = f'script_runtime_{datetime.now().strftime("%y%m%d_%H%M%S")}.json'
                out_path = os.path.join(out_dir, out_name)
                with open(out_path, 'w', encoding='utf-8') as sf:
                    json.dump(script_info, sf, ensure_ascii=False, indent=2)
                print('Saved script runtime to', out_path)
            except Exception:
                try:
                    fallback = tempfile.gettempdir()
                    out_name = f'script_runtime_{datetime.now().strftime("%y%m%d_%H%M%S")}.json'
                    out_path = os.path.join(fallback, out_name)
                    with open(out_path, 'w', encoding='utf-8') as sf:
                        json.dump(script_info, sf, ensure_ascii=False, indent=2)
                    print('Saved script runtime to', out_path)
                except Exception:
                    print('Failed to save script runtime summary.')
        except Exception:
            pass
