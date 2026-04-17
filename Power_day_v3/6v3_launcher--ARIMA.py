"""CLI launcher: pass command-line arguments through to 6v3-autoTs_WeatherToDayWh.py

Usage examples:
  python 6v3_launcher.py --default_max_generations 5 --horizons 30,14,7
  python 6v3_launcher.py --default_model_list '["ARIMA","RandomForest"]' --InfiniteLoop true

This script only forwards the supported flags to the target script and runs it
with the same Python interpreter. It does not modify the target file.
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
horizons = [120, 90, 60, 30]
# horizons = [90, 60, 30]
# horizons = [9, 6, 3]
# horizons = [9]
# horizons = [3, 2, 1]
# horizons = [1]
# horizons = [1]

# 範例：InfiniteLoop = False
# InfiniteLoop = True
InfiniteLoop = False

# 範例：default_model_list = ['ARIMA', 'RandomForest']
# default_model_list = None
# default_model_list = 'superFast' # 預設模型組合（AutoTS 內建的快速測試組合）
# default_model_list = ['LSTM'] # 只測試 LSTM
default_model_list = ['ARIMA'] # 只測試 ARIMA

# 範例：default_ensemble = ['auto','simple']
# default_ensemble = None
default_ensemble = ['simple'] # 預設 ensemble 方法（AutoTS 內建的簡單平均）
# default_ensemble = ['auto','simple', 'DistanceWeightedEnsemble', 'weighted_ensemble'] # 預設 ensemble 方法（AutoTS 內建的簡單平均）
# default_ensemble = ['auto','simple']
# default_ensemble = ['simple'] # 預設 ensemble 方法（AutoTS 內建的簡單平均）
# 範例：default_n_jobs = 2
# default_n_jobs = None
# default_n_jobs = 4 # 使用 4 個 CPU 核心
default_n_jobs = -1 # 使用全部 CPU 核心


# 範例：default_transformer_list = ['DifferencedTransformer','Scaler']
# default_transformer_list = None
# default_transformer_list = 'default' # 預設前處理組合（AutoTS 內建的預設組合）,
# default_transformer_list = []
default_transformer_list = [
    "DifferencedTransformer", # 避免被「抹平」成水平線
    "Scaler", # 避免被「抹平」成水平線
    ## LSTM 常見的前處理（不一定適合本資料集，請自行評估）
    'MinMaxScaler',       # LSTM 必備
    'Detrend',            # 去趨勢
    'DatepartRegression', # 加入時間特徵（小時、星期、季節
    ]

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

# 範例：random_seed = 12345
random_seed = None
# random_seed = 12345

HERE = os.path.dirname(__file__)
TARGET = os.path.join(HERE, '6v3-autoTs_WeatherToDayWh.py')

if not os.path.exists(TARGET):
    print(f'Error: target script not found: {TARGET}')
    sys.exit(2)

SUPPORTED = [
    'default_max_generations', 'horizons', 'InfiniteLoop', 'default_model_list',
    'default_ensemble', 'default_n_jobs', 'default_transformer_list', 'default_num_validations',
    'enable_future_regressor', 'enable_fit_future_regressor', 'enable_predict_future_regressor', 'random_seed'
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

def main():
    parser = argparse.ArgumentParser(description='Forward CLI args to 6v3-autoTs_WeatherToDayWh.py')
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

    forward = build_forward_args(args)
    # append any remaining extras (they may include flags for the target)
    if args.extra:
        # args.extra may start with '--' or be a list; extend directly
        forward.extend(args.extra)

    cmd = [sys.executable, TARGET] + forward
    print('Running:', ' '.join(shlex.quote(p) for p in cmd))
    try:
        rc = subprocess.call(cmd)
        sys.exit(rc)
    except KeyboardInterrupt:
        print('Interrupted')
        sys.exit(1)

if __name__ == '__main__':
    main()
