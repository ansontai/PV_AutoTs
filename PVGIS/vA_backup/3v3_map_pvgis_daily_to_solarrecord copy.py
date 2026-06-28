"""Launcher: 用指定的 PVGIS raw 檔產生 daily，並呼叫 mapping 腳本輸出 SolarRecord-ready CSV

用法範例:
  python 3v3-map_pvgis_daily_to_solarrecord.py --raw raw/PVGIS/raw/tmy_24.148_120.703_2005_2023.csv
"""
from pathlib import Path
import shutil
import subprocess
import sys
import time
import argparse

HERE = Path(__file__).parent
RAW_DIR = HERE / 'raw'
OUTPUT_DIR = HERE / 'output'

# aggregator script (3v2) 會尋找並讀取固定檔名，這裡以該預期檔名為目標
EXPECTED_RAW_BASENAME = 'Timeseries_24.148_120.703_E5_0kWp_crystSi_25_35deg_1deg_2005_2005.csv'
AGG_SCRIPT = HERE / '3v2-PVGIS_TimeseriesCsv_hourly_to_daily_And_MappingToMySolarRecord.py'
MAP_SCRIPT = HERE / 'map_pvgis_daily_to_solarrecord.py'


def find_latest_daily(output_dir: Path):
    files = list(output_dir.glob('*daily*.csv'))
    if not files:
        files = list(output_dir.glob('*.csv'))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def run(cmd, **kwargs):
    print('> ', ' '.join(map(str, cmd)))
    subprocess.run(cmd, check=True, **kwargs)


def main():
    p = argparse.ArgumentParser(description='Launcher: PVGIS raw -> daily -> solarrecord-ready')
    p.add_argument('--raw', type=Path, required=True, help='raw input PVGIS file (e.g. raw/PVGIS/raw/tmy_...csv)')
    p.add_argument('--no-agg', action='store_true', help='跳過 aggregator，直接使用已存在 daily CSV')
    p.add_argument('--force', action='store_true', help='覆寫 aggregator 的預期 raw 檔')
    p.add_argument('--preview-map', action='store_true', help='在 mapping 階段使用 --preview 模式')
    args = p.parse_args()

    raw_path = args.raw
    if not raw_path.exists():
        print('指定 raw 檔不存在:', raw_path)
        sys.exit(2)

    expected_raw = RAW_DIR / EXPECTED_RAW_BASENAME
    backup_path = None

    try:
        if not args.no_agg:
            # 若預期檔已存在，備份後再覆寫（除非 user 保留）
            if expected_raw.exists():
                if args.force:
                    ts = int(time.time())
                    backup_path = expected_raw.with_suffix(f'.bak.{ts}')
                    shutil.move(expected_raw, backup_path)
                    print('已備份原預期 raw 檔到', backup_path)
                else:
                    print('預期 raw 檔已存在:', expected_raw)
                    print('使用 --force 覆寫或先移除該檔後再執行')
                    sys.exit(3)

            # 複製使用者提供的 raw 檔到 aggregator 預期位置
            expected_raw.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw_path, expected_raw)
            print('已複製 raw 到預期檔名:', expected_raw)

            # 執行 aggregator
            if not AGG_SCRIPT.exists():
                print('找不到 aggregator 腳本:', AGG_SCRIPT)
                sys.exit(4)
            run([sys.executable, str(AGG_SCRIPT)])

        # 找最近的 daily CSV
        daily = find_latest_daily(OUTPUT_DIR)
        if daily is None:
            print('找不到 daily CSV，請確認 aggregator 已產生 output 檔案於', OUTPUT_DIR)
            sys.exit(5)
        print('使用 daily 檔:', daily)

        # 執行 mapping 腳本
        if not MAP_SCRIPT.exists():
            print('找不到 mapping 腳本:', MAP_SCRIPT)
            sys.exit(6)
        map_cmd = [sys.executable, str(MAP_SCRIPT), '--input', str(daily)]
        if args.preview_map:
            map_cmd.append('--preview')
        else:
            # 預設輸出放在同一 output 資料夾
            out_path = OUTPUT_DIR / (daily.stem + '-solarrecord-ready.csv')
            if out_path.exists():
                if args.force:
                    out_path.unlink()
                else:
                    print('mapping 輸出檔已存在:', out_path, '使用 --force 覆蓋')
                    sys.exit(7)
            map_cmd += ['--output', str(out_path)]

        run(map_cmd)

    finally:
        # 還原備份的原始預期檔（若有備份）
        if backup_path and backup_path.exists():
            if expected_raw.exists():
                expected_raw.unlink()
            shutil.move(backup_path, expected_raw)
            print('已還原原始預期 raw 檔到', expected_raw)


if __name__ == '__main__':
    main()
import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from datetime import datetime


HERE = Path(__file__).parent


def find_default_input(output_dir: Path):
    if not output_dir.exists():
        return None
    for p in sorted(output_dir.glob('*.csv')):
        if 'daily' in p.name.lower():
            return p
    # fallback: first csv
    files = list(output_dir.glob('*.csv'))
    return files[0] if files else None


def to_kwh(series, unit_hint=None):
    # If series appears to be Wh (large values) convert to kWh
    s = pd.to_numeric(series, errors='coerce')
    if unit_hint == 'Wh' or (s.dropna().median() is not np.nan and s.dropna().median() > 100):
        return s / 1000.0
    return s


MAPPING_PRIORITY = {
    'E_day_kWh': ['E_day_kWh', 'P_kWh', 'P_Wh'],
    'P_kWh': ['P_kWh', 'E_day_kWh', 'P_Wh'],
    'Hpoa_day_kWhm2': ['Hpoa_day_kWhm2', 'G(i)_kWhm2', 'G(i)_Whm2'],
    'P_mean': ['P_mean'],
    'T2m_mean': ['T2m_mean'],
    'RH_mean': ['RH_mean'],
    'WS10m_mean': ['WS10m_mean'],
    'WD10m_circmean': ['WD10m_circmean', 'WD10m'],
    'n_obs_P': ['n_obs_P'],
    'valid_frac_P': ['valid_frac_P'],
}


def map_columns(df: pd.DataFrame, fill_zero: bool = False):
    out = pd.DataFrame()

    # ensure date exists and normalized
    if 'date' in df.columns:
        out['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    elif df.index.dtype.kind in 'M':
        out['date'] = pd.to_datetime(df.index).normalize().strftime('%Y-%m-%d')
    else:
        raise SystemExit('找不到 date 欄位，請確認輸入 daily CSV 含有 `date` 或可轉為索引的時間欄位')

    missing = []
    for target, candidates in MAPPING_PRIORITY.items():
        found = False
        for c in candidates:
            if c in df.columns:
                if c.endswith('_Whm2') or c.endswith('_Wh'):
                    # convert Wh -> kWh where appropriate
                    out[target] = to_kwh(df[c], unit_hint='Wh')
                else:
                    out[target] = pd.to_numeric(df[c], errors='coerce')
                found = True
                break
        if not found:
            out[target] = np.nan
            missing.append(target)

    # if E_day_kWh exists but P_kWh missing, copy
    if 'E_day_kWh' in out.columns and pd.notna(out['E_day_kWh']).any() and out['P_kWh'].isna().all():
        out['P_kWh'] = out['E_day_kWh']

    # ensure types
    for col in out.columns:
        if col != 'date':
            out[col] = pd.to_numeric(out[col], errors='coerce')

    # derive valid_frac_P from n_obs_P if missing
    if 'valid_frac_P' in out.columns and out['valid_frac_P'].isna().all() and 'n_obs_P' in out.columns:
        out['valid_frac_P'] = out['n_obs_P'] / 24.0

    if fill_zero:
        out = out.fillna(0)

    return out, missing


def main():
    p = argparse.ArgumentParser(description='Map PVGIS daily CSV to SolarRecord-style CSV')
    p.add_argument('--input', '-i', type=Path, help='input daily CSV (from PVGIS aggregator)')
    p.add_argument('--output', '-o', type=Path, help='output CSV path')
    p.add_argument('--force', action='store_true', help='overwrite output if exists')
    p.add_argument('--preview', action='store_true', help='show preview and exit')
    p.add_argument('--fill-zero', action='store_true', help='fill missing numeric with zero')
    args = p.parse_args()

    input_path = args.input or find_default_input(HERE / 'output')
    if input_path is None:
        print('找不到預設輸入檔，請使用 --input 指定 daily CSV', file=sys.stderr)
        sys.exit(2)

    if not input_path.exists():
        print(f'輸入檔不存在: {input_path}', file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(input_path)

    mapped, missing = map_columns(df, fill_zero=args.fill_zero)

    if args.preview:
        print('來源檔:', input_path)
        print('欄位對齊結果：')
        for k in mapped.columns:
            print(' -', k)
        if missing:
            print('\n缺少的目標欄位（已填 NaN）：', missing)
        print('\n前 5 列預覽：')
        print(mapped.head().to_string(index=False))
        return

    out_path = args.output or (HERE / 'output' / (input_path.stem + '-solarrecord-ready.csv'))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.force:
        print(f'輸出檔已存在: {out_path}（使用 --force 覆蓋）', file=sys.stderr)
        sys.exit(3)

    # write metadata comment as first line (not breaking standard CSV reading by pandas)
    header_note = f'# generated_by: map_pvgis_daily_to_solarrecord.py at {datetime.utcnow().isoformat()}Z from {input_path.name}'
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(header_note + '\n')
        mapped.to_csv(fh, index=False)

    print('已輸出:', out_path)


if __name__ == '__main__':
    main()
