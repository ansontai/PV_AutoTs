#!/usr/bin/env python3
"""fix_timestamps.py

簡單工具：檢查 CSV 中的時間欄位，標記/移除或修正「年份/時間異常」列。

用法:
  python csv/fix_timestamps.py path/to/file.csv --mode drop
  python csv/fix_timestamps.py path/to/file.csv --mode fix-year

模式說明:
  drop     : 移除被判為異常的列，輸出為原檔名 + "-cleaned.csv"
  fix-year : 將異常列的年份改為資料中主要年份（mode），輸出為原檔名 + "-fixed-year.csv"

此腳本會嘗試自動找出時間欄位，若找不到請手動指定欄名（小改程式即可）。
"""
import argparse
from pathlib import Path
import sys
import pandas as pd
from dateutil import parser as dateutil_parser
import os
import tempfile

# --- 可於此修改輸入/輸出路徑常數，方便後續快速調整 ---
# 預設輸入檔（可被命令列參數覆寫）
INPUT_CSV = Path("csv/SolarRecord_260310_1829-row-number.csv")
# 預設輸出修正後主檔與變更紀錄檔
OUTPUT_FIXED = Path("csv/SolarRecord_260310_1829-fixed_0b.csv")
OUTPUT_CHANGES = Path("csv/SolarRecord_260310_1829-fixed_0b-changes.csv")

COMMON_TIME_NAMES = [
    "datetime",
    "date",
    "time",
    "timestamp",
    "local_time",
    "localtime",
    "Date",
    "Time",
]


def find_time_col(df: pd.DataFrame):
    for c in df.columns:
        if c in COMMON_TIME_NAMES:
            return c
    # try case-insensitive match containing keywords
    lowcols = {c.lower(): c for c in df.columns}
    for key in ["date", "time", "datetime", "timestamp", "local"]:
        for lc, orig in lowcols.items():
            if key in lc:
                return orig
    return None


def robust_parse_one(s):
    s = str(s).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return pd.NaT
    # 1) 先用 pandas 嘗試（快速）
    try:
        dt = pd.to_datetime(s, errors="raise")
        return dt
    except Exception:
        pass
    # 2) 純數字：試 epoch (秒) 或 Excel 序列
    if s.replace('.', '', 1).isdigit():
        try:
            return pd.to_datetime(int(float(s)), unit="s")
        except Exception:
            try:
                return pd.to_datetime("1899-12-30") + pd.to_timedelta(int(float(s)), unit="D")
            except Exception:
                pass
    # 3) 用 dateutil 作最後嘗試
    try:
        return dateutil_parser.parse(s, dayfirst=False, fuzzy=True)
    except Exception:
        return pd.NaT


def safe_write_csv(df, path: Path):
    # 先嘗試寫入暫存檔再原子替換，若失敗則寫入帶 timestamp 的備援檔
    try:
        tmp = Path(str(path) + ".tmp")
        df.to_csv(tmp, index=False)
        try:
            tmp.replace(path)
        except PermissionError:
            alt = path.with_name(path.stem + "-writen-fallback-" + pd.Timestamp.now().strftime("%Y%m%d%H%M%S") + path.suffix)
            tmp.replace(alt)
            print(f"無法覆寫 {path}，已寫入備援檔: {alt}")
    except PermissionError:
        alt = path.with_name(path.stem + "-writen-fallback-" + pd.Timestamp.now().strftime("%Y%m%d%H%M%S") + path.suffix)
        df.to_csv(alt, index=False)
        print(f"PermissionError: 直接寫入備援檔: {alt}")


def main():
    p = argparse.ArgumentParser(description="Fix rows with anomalous timestamps")
    p.add_argument("csv", nargs='?', default=str(INPUT_CSV), help="input CSV file path (optional, default from top-level variable)")
    args = p.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print("Error: file not found:", path, file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(path, low_memory=False)
    col = find_time_col(df)
    if col is None:
        print("找不到時間欄位。請檢查欄名或手動修改腳本以指定欄位。可用欄位：", list(df.columns))
        sys.exit(2)

    print(f"偵測到時間欄位: {col}")
    df["__dt"] = pd.to_datetime(df[col], errors="coerce")
    # 對解析失敗的列再用更健壯的解析器嘗試
    mask_na = df["__dt"].isna()
    if mask_na.any():
        df.loc[mask_na, "__dt"] = df.loc[mask_na, col].apply(robust_parse_one)
    n_null = df['__dt'].isna().sum()
    if n_null:
        print(f"注意：有 {n_null} 筆無法解析為時間（會被視為異常）")

    # 以主要年份判定異常（mode 年）
    years = df['__dt'].dropna().dt.year
    if years.empty:
        print("無任何可解析的時間，停止。")
        sys.exit(1)

    main_year = int(years.mode().iloc[0])
    print(f"資料主要年份（mode）: {main_year}")

    # 只有年份 > 2060 或解析失敗才視為異常
    mask_bad = (df['__dt'].isna()) | (df['__dt'].dt.year > 2060)
    n_bad = mask_bad.sum()
    print(f"偵測到 {n_bad} 筆可能的異常時間列（包含解析失敗）")

    # 固定執行修正流程（fix-year）
    out = OUTPUT_FIXED
    changes_out = OUTPUT_CHANGES

    # 用位置索引方便處理，並保留原始資訊以便比對/輸出
    df2 = df.reset_index().copy()    # 原 index 存在欄位 'index'
    df2['__orig_time_str'] = df2[col].astype(str)
    df2['__orig_dt'] = df2['__dt']  # 保存修改前的解析結果

    # 判為異常（解析失敗 或 年份大於 2060）
    bad = (df2['__dt'].isna()) | (df2['__dt'].dt.year > 2060)
    if not bad.any():
        df2 = df2.drop(columns=["__orig_dt", "__orig_time_str"])
        safe_write_csv(df2, out)
        # 輸出空的 changes 檔（方便流程自動化）
        pd.DataFrame(columns=df.columns.tolist() + ["__orig_time_str", "fixed_time", "index"]).to_csv(changes_out, index=False)
        print(f"沒有需要修正的列，輸出複本: {out}")
        return

    # 找出連續區段（只處理 bad=True 的 group）
    group_id = (bad != bad.shift(fill_value=False)).cumsum()
    for gid, group_df in df2[bad].groupby(group_id):
        first_pos = group_df.index[0]
        prev_pos = first_pos - 1

        # 決定基準時間（prev_row.__dt + 1 minute），若 prev 無效或不存在則 fallback 到 main_year-01-01 00:00
        if prev_pos >= 0:
            prev_dt = df2.at[prev_pos, '__dt']
            if pd.isna(prev_dt):
                base_first = pd.Timestamp(year=main_year, month=1, day=1, hour=0, minute=0)
            else:
                base_first = prev_dt + pd.Timedelta(minutes=1)
        else:
            base_first = pd.Timestamp(year=main_year, month=1, day=1, hour=0, minute=0)

        # 原始第一列時間（可能為 NaT）
        orig_first = group_df['__dt'].iloc[0]

        # 若原始第一列可解析，保留每列相對於 orig_first 的時間差；否則以逐列 +1 分鐘備援
        if pd.notna(orig_first):
            for idx in group_df.index:
                orig_dt = df2.at[idx, '__dt']
                if pd.notna(orig_dt):
                    delta = orig_dt - orig_first
                    df2.at[idx, '__dt'] = base_first + delta
                else:
                    # 若某列原本無時間，衍生策略：以前一列的新時間 +1 分鐘（若可用），否則以 base_first + offset
                    prev_idx = idx - 1
                    if prev_idx >= 0 and pd.notna(df2.at[prev_idx, '__dt']):
                        df2.at[idx, '__dt'] = df2.at[prev_idx, '__dt'] + pd.Timedelta(minutes=1)
                    else:
                        offset = list(group_df.index).index(idx)
                        df2.at[idx, '__dt'] = base_first + pd.Timedelta(minutes=offset)
        else:
            # 原始第一列無法解析 -> 以 base_first 並每列加 0/1/2... 分鐘
            for offset, idx in enumerate(group_df.index):
                df2.at[idx, '__dt'] = base_first + pd.Timedelta(minutes=offset)

    # 新時間字串
    df2['fixed_time'] = df2['__dt'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # 判別哪些列真正被修改（原解析值與新解析值不同，或原解析為 NaT 但現在有值）
    mask_changed = (
        (df2['__orig_dt'].isna() & df2['__dt'].notna())
        | (df2['__orig_dt'].notna() & (df2['__orig_dt'] != df2['__dt']))
    )

    # 建立變更紀錄：包含原始欄位、原始字串、修正後時間、以及原始 row index（reset 前的 index 存在 'index' 欄）
    changed_cols = df.columns.tolist() + ['__orig_time_str', 'fixed_time', 'index']
    changes_df = df2.loc[mask_changed, changed_cols]
    # 寫出變更紀錄檔
    safe_write_csv(changes_df, changes_out)

    # 輸出整個修正後的檔案（把時間欄覆寫為修正後的時間字串）
    df_out = df2.copy()
    # 覆寫原時間欄為修正後字串
    df_out[col] = df2['__dt'].dt.strftime('%Y-%m-%d %H:%M:%S')
    # 新增一欄 _dt 儲存解析後的時間（同樣格式），方便下游使用
    # df_out['_dt'] = df2['__dt'].dt.strftime('%Y-%m-%d %H:%M:%S')
    # 移除內部輔助欄
    df_out = df_out.drop(columns=['__orig_dt', '__orig_time_str', 'fixed_time', 'index'], errors='ignore')
    # 只保留使用者指定的欄位（存在的才保留，順序依使用者要求）
    desired = [
        'LocalTime','W','DHT11_temp','DHT11_humidity','LM35_tempC',
        'INA219_busVoltage_V','INA219_shuntVoltage_mV','INA219_current_mA',
        'INA219_power_mW','ACS712_20A_current_A','_dt'
    ]
    available = [c for c in desired if c in df_out.columns]
    df_out = df_out[available]
    safe_write_csv(df_out, out)

    print(f"已依規則修正連續異常區段並輸出: {out}")
    print(f"修改紀錄已輸出: {changes_out}")


if __name__ == '__main__':
    main()
