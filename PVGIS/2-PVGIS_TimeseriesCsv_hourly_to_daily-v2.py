import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# 檔案路徑（相對於本檔案所在目錄）
HERE = Path(__file__).parent
DEFAULT_INPUT = HERE / "output" / "Timeseries_24.148_120.703_E5_0kWp_crystSi_25_35deg_1deg_2005_2005[P_scaled_hourly].csv"
DEFAULT_INPUT_RAW = HERE / "raw" / "Timeseries_24.148_120.703_E5_0kWp_crystSi_25_35deg_1deg_2005_2005[P_scaled_hourly].csv"
DEFAULT_OUTPUT = HERE / "output" / "Timeseries_24.148_120.703_E5_0kWp_crystSi_25_35deg_1deg_2005_2005[UTC+8][daily][scaled].csv"

parser = argparse.ArgumentParser(description='Convert PVGIS timeseries hourly CSV to daily aggregates.')
parser.add_argument('--input', '-i', type=Path, default=None, help='Input CSV path')
parser.add_argument('--output', '-o', type=Path, default=DEFAULT_OUTPUT, help='Output CSV path')
args = parser.parse_args()

if args.input is None:
    if DEFAULT_INPUT.exists():
        INPUT = DEFAULT_INPUT
    elif DEFAULT_INPUT_RAW.exists():
        INPUT = DEFAULT_INPUT_RAW
    else:
        raise FileNotFoundError(f"No input file found. Searched: {DEFAULT_INPUT}, {DEFAULT_INPUT_RAW}")
else:
    INPUT = args.input
OUTPUT = args.output

# PVGIS 時序檔在最前面有 metadata（非 CSV 欄位），找出真正的 header 行並跳過前面的說明文字
from io import StringIO

with open(INPUT, "r", encoding="utf-8") as fh:
    lines = fh.readlines()

header_row = None
for i, line in enumerate(lines):
    if line.lstrip().startswith("time,"):
        header_row = i
        break
if header_row is None:
    raise SystemExit("未在輸入檔找到 'time,' 欄位標頭，無法解析")

import re

# 只取從 header 行開始、且符合時間序列格式的資料區段（避免後面的註解或說明造成解析錯誤）
data_lines = []
for i, line in enumerate(lines[header_row:]):
    if i == 0:
        data_lines.append(line)
        continue
    if re.match(r"^(?:\d{8}:\d{4}|\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),", line):
        data_lines.append(line)
        continue
    # 空白行容許，否則視為資料區段結束
    if line.strip() == "":
        break
    # 若遇到非資料行（例如描述文字），則結束解析
    break

csv_text = "".join(data_lines)
df = pd.read_csv(StringIO(csv_text))

# 解析 PVGIS time 欄，可能是 YYYYMMDD:HHMM 或 ISO8601（例如已被 upstream 轉換）
df["time_utc"] = pd.to_datetime(df["time"], format="%Y%m%d:%H%M", utc=True, errors="coerce")
if df["time_utc"].isna().all():
    df["time_utc"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

if df["time_utc"].isna().all():
    raise ValueError("無法解析 time 欄位，請檢查輸入格式")

# 轉成本地時間（台灣 UTC+8）；如果你想用 UTC 日界線，就跳過這行
df["time_local"] = df["time_utc"].dt.tz_convert("Asia/Taipei")

df = df.set_index("time_local").sort_index()

# 自動估 Δt（小時）
dt_hours = df.index.to_series().diff().dt.total_seconds().median() / 3600.0

# 逐時功率/輻照度 -> 能量/輻照量（kWh, kWh/m2）
df["E_kWh"] = df["P"] * dt_hours / 1000.0
df["Hpoa_kWhm2"] = df["G(i)"] * dt_hours / 1000.0


def circular_mean_deg(deg, w=None):
    if deg is None:
        return np.nan
    deg = deg.dropna()
    if deg.empty:
        return np.nan
    rad = np.deg2rad(deg.to_numpy())
    if w is None:
        w_arr = np.ones_like(rad)
    else:
        try:
            w_arr = np.asarray(w.loc[deg.index])
        except Exception:
            w_arr = np.asarray(w)
        if np.all(w_arr == 0):
            w_arr = np.ones_like(w_arr)
    x = np.sum(w_arr * np.cos(rad))
    y = np.sum(w_arr * np.sin(rad))
    ang = np.arctan2(y, x)
    return (np.rad2deg(ang) + 360) % 360


def circular_std_deg(deg, w=None):
    if deg is None:
        return np.nan
    deg = deg.dropna()
    if deg.empty:
        return np.nan
    rad = np.deg2rad(deg.to_numpy())
    if w is None:
        w_arr = np.ones_like(rad)
    else:
        try:
            w_arr = np.asarray(w.loc[deg.index])
        except Exception:
            w_arr = np.asarray(w)
        if np.all(w_arr == 0):
            w_arr = np.ones_like(w_arr)
    x = np.sum(w_arr * np.cos(rad))
    y = np.sum(w_arr * np.sin(rad))
    R = np.sqrt(x * x + y * y) / np.sum(w_arr)
    R = np.clip(R, 1e-12, 1.0)
    std_rad = np.sqrt(-2.0 * np.log(R))
    return np.rad2deg(std_rad)


# 建立完整的日聚合欄位（命名一致：_mean/_min/_max/_std/_p10/_p90/_Whm2/_kWhm2/_Wh/_kWh）
counts_per_day = df.resample('D').size()
daily = pd.DataFrame(index=counts_per_day.index)

# 保留舊有的 kWh 日合計輸出
daily['E_day_kWh'] = df['E_kWh'].resample('D').sum()
daily['Hpoa_day_kWhm2'] = df['Hpoa_kWhm2'].resample('D').sum()

# 輻照 (Wh/m2) 與統計
irr_cols = ["G(i)", "G(h)", "Gb(n)", "Gd(h)", "IR(h)"]
for c in irr_cols:
    if c in df.columns:
        daily[c + '_Whm2'] = (df[c] * dt_hours).resample('D').sum()
        daily[c + '_kWhm2'] = daily[c + '_Whm2'] / 1000.0
        daily[c + '_mean'] = df[c].resample('D').mean()
        daily[c + '_min'] = df[c].resample('D').min()
        daily[c + '_max'] = df[c].resample('D').max()
        daily[c + '_std'] = df[c].resample('D').std()
        daily[c + '_p10'] = df[c].resample('D').quantile(0.1)
        daily[c + '_p90'] = df[c].resample('D').quantile(0.9)

# 功率/能量類欄位 -> 能量與統計（支援額外 P 欄位）
power_cols = [c for c in ['P', 'P_mapped_Wh', 'P_normalized_0_1_Wh'] if c in df.columns]
for c in power_cols:
    # 每筆量為 value * dt_hours -> Wh
    daily[f'{c}'] = (df[c] * dt_hours).resample('D').sum()
    daily[f'{c}_kWh'] = daily[f'{c}'] / 1000.0
    # 保持原有命名風格（mean/min/max/std/p10/p90）以維持相容性
    daily[f'{c}_mean'] = df[c].resample('D').mean()
    daily[f'{c}_min'] = df[c].resample('D').min()
    daily[f'{c}_max'] = df[c].resample('D').max()
    daily[f'{c}_std'] = df[c].resample('D').std()
    daily[f'{c}_p10'] = df[c].resample('D').quantile(0.1)
    daily[f'{c}_p90'] = df[c].resample('D').quantile(0.9)

# 氣象統計
if 'T2m' in df.columns:
    daily['T2m_mean'] = df['T2m'].resample('D').mean()
    daily['T2m_min'] = df['T2m'].resample('D').min()
    daily['T2m_max'] = df['T2m'].resample('D').max()
    daily['T2m_std'] = df['T2m'].resample('D').std()
    daily['T2m_p10'] = df['T2m'].resample('D').quantile(0.1)
    daily['T2m_p90'] = df['T2m'].resample('D').quantile(0.9)

if 'RH' in df.columns:
    daily['RH_mean'] = df['RH'].resample('D').mean()
    daily['RH_min'] = df['RH'].resample('D').min()
    daily['RH_max'] = df['RH'].resample('D').max()
    daily['RH_std'] = df['RH'].resample('D').std()
    daily['RH_p10'] = df['RH'].resample('D').quantile(0.1)
    daily['RH_p90'] = df['RH'].resample('D').quantile(0.9)

if 'WS10m' in df.columns:
    daily['WS10m_mean'] = df['WS10m'].resample('D').mean()
    daily['WS10m_max'] = df['WS10m'].resample('D').max()
    daily['WS10m_std'] = df['WS10m'].resample('D').std()
    daily['WS10m_p10'] = df['WS10m'].resample('D').quantile(0.1)
    daily['WS10m_p90'] = df['WS10m'].resample('D').quantile(0.9)

if 'SP' in df.columns:
    daily['SP_mean'] = df['SP'].resample('D').mean()

# 太陽高度 / 日照
if 'H_sun' in df.columns:
    daily['H_sun_max'] = df['H_sun'].resample('D').max()
    daily['H_sun_mean'] = df['H_sun'].resample('D').mean()

# 風向圓形統計
if 'WD10m' in df.columns:
    def _wd_circ_mean(s):
        w = df['WS10m'] if 'WS10m' in df.columns else None
        return circular_mean_deg(s, w=w)
    def _wd_circ_std(s):
        w = df['WS10m'] if 'WS10m' in df.columns else None
        return circular_std_deg(s, w=w)
    def _wd_mode_sector(s):
        degs = s.dropna()
        if degs.empty:
            return np.nan
        sectors = ((degs + 22.5) // 45).astype(int) % 8
        labels = ['N','NE','E','SE','S','SW','W','NW']
        mode_idx = sectors.value_counts().idxmax()
        return labels[int(mode_idx)]
    daily['WD10m_circmean'] = df['WD10m'].resample('D').apply(_wd_circ_mean)
    daily['WD10m_circstd_deg'] = df['WD10m'].resample('D').apply(_wd_circ_std)
    daily['WD10m_mode_sector'] = df['WD10m'].resample('D').apply(_wd_mode_sector)

# 觀測數與有效比例
key_cols = [c for c in ['P','P_mapped_Wh','P_normalized_0_1_Wh','G(i)','G(h)','T2m','RH','WS10m','WD10m','SP','H_sun'] if c in df.columns]
for c in key_cols:
    n_obs = df[c].resample('D').count()
    daily[f'n_obs_{c}'] = n_obs
    with np.errstate(divide='ignore', invalid='ignore'):
        daily[f'valid_frac_{c}'] = n_obs / counts_per_day.replace(0, np.nan)

# index normalize 並加入 date 欄 (YYYY-MM-DD) 以及 doy/month/day 欄位
daily.index = daily.index.normalize()
try:
    daily['date'] = daily.index.strftime('%Y-%m-%d')
except Exception:
    daily['date'] = pd.to_datetime(daily.index).strftime('%Y-%m-%d')

daily['doy'] = daily.index.dayofyear
daily['month'] = daily.index.month
daily['day'] = daily.index.day

cols = ['date', 'doy', 'month', 'day'] + [c for c in daily.columns if c not in {'date', 'doy', 'month', 'day'}]
daily = daily[cols]

# 確保輸出目錄存在，然後寫檔
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
daily.to_csv(OUTPUT, index=False)