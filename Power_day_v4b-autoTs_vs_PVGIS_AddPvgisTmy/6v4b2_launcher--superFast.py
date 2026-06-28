"""CLI launcher: pass command-line arguments through to 6v4b2-autoTs_WeatherToDayWh.py

TWO-PHASE EXECUTION:
  Phase 1: Standard training with specified horizons (default: 90d)
  Phase 2: 360-day forecast with future regressors (TMY) enabled

Usage examples:
  python 6v4b2_launcher--superFast.py
  python 6v4b2_launcher--superFast.py --horizons 30,60
  python 6v4b2_launcher--superFast.py --default_model_list '["ARIMA","RandomForest"]'
  python 6v4b2_launcher--superFast.py --no_360d  (skip 360d forecast)
"""
from __future__ import annotations
import argparse
import os
import shlex
import subprocess
import sys
import json


# --- 在此區編輯參數 (設為 None 表示不覆寫) ---
# 範例：horizons = [30, 14, 7]
# horizons = None
# horizons = [90, 60, 30]
horizons = [90]
# horizons = [9, 6, 3]
# horizons = [9]
# horizons = [3, 2, 1]
# horizons = [3, 2]
# horizons = [1]

# 範例：InfiniteLoop = False
# InfiniteLoop = True
InfiniteLoop = False

# 範例：default_model_list = ['ARIMA', 'RandomForest']
# default_model_list = None
# default_model_list = 'superFast' # 預設模型組合（AutoTS 內建的快速測試組合）
# default_model_list = ['LSTM'] # 只測試 LSTM
default_model_list = ['ARIMA'] # 只測試 ARIMA
# default_model_list = ['Prophet'] # 只測試 Prophet
# default_model_list = 'superFast' # 預設模型組合（AutoTS 內建的快速測試組合）


# 範例：default_ensemble = ['auto','simple']
# default_ensemble = None
# default_ensemble = ['simple'] # 預設 ensemble 方法（AutoTS 內建的簡單平均）
# default_ensemble = ['auto','simple', 'DistanceWeightedEnsemble', 'weighted_ensemble'] # 預設 ensemble 方法（AutoTS 內建的簡單平均）
# default_ensemble = ['auto','simple']
default_ensemble = ['simple'] # 預設 ensemble 方法（AutoTS 內建的簡單平均）


# 範例：default_n_jobs = 2
# default_n_jobs = None
# default_n_jobs = 4 # 使用 4 個 CPU 核心
default_n_jobs = -1 # 使用全部 CPU 核心


# 範例：default_transformer_list = ['DifferencedTransformer','Scaler']
# default_transformer_list = None
# default_transformer_list = 'default' # 預設前處理組合（AutoTS 內建的預設組合）,
default_transformer_list = 'superFast' # 預設前處理組合（AutoTS 內建的快速測試組合）,
# default_transformer_list = []
# default_transformer_list = [
#     "DifferencedTransformer", # 避免被「抹平」成水平線
#     "Scaler", # 避免被「抹平」成水平線
#     ## LSTM 常見的前處理（不一定適合本資料集，請自行評估）
#     'MinMaxScaler',       # LSTM 必備
#     'Detrend',            # 去趨勢
#     'DatepartRegression', # 加入時間特徵（小時、星期、季節
#     ]

# 範例：default_max_generations = 5
# default_max_generations = 15
default_max_generations = 1

# 範例：default_num_validations = 3
# default_num_validations = None
default_num_validations = 3

# 範例： enable_future_regressor = True
# enable_future_regressor = True
enable_future_regressor = False

# fine-grained future_regressor control for fit / predict phases
enable_fit_future_regressor = False
enable_predict_future_regressor = False

# 範例：enable_tmy = True # 預測時自動載入 TMY 氣象資料作為 future_regressor
# enable_tmy = True # 預測時自動載入 TMY 氣象資料作為 future_regressor
# enable_tmy = False # 預測時自動載入 TMY 氣象資料作為 future_regressor
enable_tmy = True
# 範例：random_seed = 12345
random_seed = None
# random_seed = 12345

# === 360d 預測設定 (訓練完成後執行) ===
# 設為 True 以啟用訓練後的 360d 長期預測（含 future_regressor 和 TMY）
RUN_360D_FORECAST_AFTER_TRAINING = True

HERE = os.path.dirname(__file__)
# TARGET = os.path.join(HERE, '6v3-autoTs_WeatherToDayWh.py')
TARGET = os.path.join(HERE, '6v4b2-autoTs_WeatherToDayWh.py')

if not os.path.exists(TARGET):
    print(f'Error: target script not found: {TARGET}')
    sys.exit(2)

SUPPORTED = [
    'default_max_generations', 'horizons', 'InfiniteLoop', 'default_model_list',
    'default_ensemble', 'default_n_jobs', 'default_transformer_list', 'default_num_validations',
    'enable_future_regressor', 'enable_fit_future_regressor', 'enable_predict_future_regressor',
    'enable_tmy', 'random_seed'
]


def build_forward_args(parsed):
    out = []
    for name in SUPPORTED:
        val = getattr(parsed, name)
        if val is None:
            continue
        # For lists supplied via multiple flags, argparse gives list; convert to comma string
        if isinstance(val, list):
            # forward lists as JSON so the target script can detect and parse them reliably
            try:
                s = json.dumps(val, ensure_ascii=False)
            except Exception:
                s = ','.join(map(str, val))
        else:
            s = str(val)
        out.append(f'--{name}')
        out.append(s)
    return out

def run_training(args):
    """Run standard training with horizons specified in args."""
    forward = build_forward_args(args)
    cmd = [sys.executable, TARGET] + forward
    print('=' * 70)
    print('PHASE 1: Standard Training')
    print('=' * 70)
    print('Running:', ' '.join(shlex.quote(p) for p in cmd))
    try:
        rc = subprocess.call(cmd)
        return rc
    except KeyboardInterrupt:
        print('\nTraining interrupted')
        return 1

def run_360d_forecast(base_args):
    """Run 360-day forecast with future regressors (TMY) enabled."""
    # Create args for 360d forecast
    forecast_args = argparse.Namespace()
    for name in SUPPORTED:
        setattr(forecast_args, name, getattr(base_args, name))
    
    # Override for 360d mode: enable future regressors and use only 360d horizon
    forecast_args.horizons = [360]
    forecast_args.enable_future_regressor = True
    forecast_args.enable_fit_future_regressor = True
    forecast_args.enable_predict_future_regressor = True
    forecast_args.enable_tmy = True
    forecast_args.InfiniteLoop = False
    # Keep other settings (model_list, ensemble, transformers, etc.) for consistency
    
    forward = build_forward_args(forecast_args)
    cmd = [sys.executable, TARGET] + forward
    print()
    print('=' * 70)
    print('PHASE 2: 360-Day Forecast with Future Regressors (TMY Data)')
    print('=' * 70)
    print('Running:', ' '.join(shlex.quote(p) for p in cmd))
    try:
        rc = subprocess.call(cmd)
        return rc
    except KeyboardInterrupt:
        print('\n360d forecast interrupted')
        return 1

def main():
    parser = argparse.ArgumentParser(
        description='Two-phase launcher: Train AutoTS, then 360d forecast with future regressors'
    )
    parser.add_argument('--default_max_generations')
    parser.add_argument('--horizons', help='comma list or JSON list')
    parser.add_argument('--InfiniteLoop')
    parser.add_argument('--default_model_list', help='comma list or JSON list')
    parser.add_argument('--default_ensemble', help='comma list or JSON list')
    parser.add_argument('--default_n_jobs')
    parser.add_argument('--default_transformer_list', help='comma list or JSON list')
    parser.add_argument('--default_num_validations')
    parser.add_argument('--enable_future_regressor')
    parser.add_argument('--enable_fit_future_regressor')
    parser.add_argument('--enable_predict_future_regressor')
    parser.add_argument('--random_seed')
    parser.add_argument('--enable_tmy')
    parser.add_argument('--run_360d', action='store_true', help='Enable 360d forecast (overrides RUN_360D_FORECAST_AFTER_TRAINING)')
    parser.add_argument('--no_360d', action='store_true', help='Disable 360d forecast')
    # allow passing arbitrary extra args which will be forwarded unchanged
    parser.add_argument('extra', nargs=argparse.REMAINDER, help='Extra args to forward')

    args = parser.parse_args()
    # If a supported option was not passed on the CLI, but the launcher
    # module defines a top-level variable for it (e.g. default_model_list),
    # use that module-level default so the launcher actually forwards the
    # intended values without requiring explicit CLI flags.
    for name in SUPPORTED:
        if getattr(args, name) is None:
            val = globals().get(name)
            if val is not None:
                setattr(args, name, val)

    # Determine if 360d forecast should run
    if args.no_360d:
        run_360d = False
    elif args.run_360d:
        run_360d = True
    else:
        run_360d = RUN_360D_FORECAST_AFTER_TRAINING

    # Phase 1: Run standard training
    print('🚀 Starting AutoTS Training + 360d Forecast Pipeline\n')
    rc_train = run_training(args)
    if rc_train != 0:
        print(f'\n❌ Standard training FAILED (exit code {rc_train})')
        sys.exit(rc_train)
    
    print('\n✅ Standard training COMPLETED successfully\n')
    
    # Phase 2: Run 360d forecast if enabled and training succeeded
    if run_360d:
        rc_360d = run_360d_forecast(args)
        if rc_360d != 0:
            print(f'\n❌ 360d forecast FAILED (exit code {rc_360d})')
            sys.exit(rc_360d)
        print('\n' + '=' * 70)
        print('✅ SUCCESS: Both training and 360d forecast completed!')
        print('=' * 70)
        print('\n📊 Output files:')
        print('  📈 360-day forecast chart: forecast_vs_actual_vs_naive_lag1_360d-format2.png')
        print('  📁 Check output/ folder for all results')
    else:
        print('ℹ️  360d forecast is disabled\n')
    
    sys.exit(0)

if __name__ == '__main__':
    main()
