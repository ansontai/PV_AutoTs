"""
將 SolarRecord(260204)_d_Wh.csv 與 202102--202602-d.csv 依照日期(LocalTime/Date)合併

輸出檔案會包含所有來自 Wh 檔的欄位，加上對應的天氣觀測欄位。
"""
import pandas as pd
# 來源檔案（以本程式所在目錄為基準）
from pathlib import Path


# Defaults (集中於檔案頂端方便修改)
DEFAULT_FILE_WH = 'SolarRecord_260310_1829-daily-1d.csv'
DEFAULT_FILE_EXT = '2000--202602-d-forWh_4b.csv'
DEFAULT_OUTPUT_SUBDIR = 'csv/output'  # 相對於 base_dir.parent
DEFAULT_OUTPUT_NAME = 'SolarRecord(260310)_d_forWh_WithCodis.csv'


# 來源檔案（以本程式所在目錄為基準）
base_dir = Path(__file__).parent
file_wh = base_dir / DEFAULT_FILE_WH
file_ext = base_dir / DEFAULT_FILE_EXT

print(f"讀取 {file_wh}")
df_wh = pd.read_csv(file_wh)
print(f"讀取 {file_ext}")
df_ext = pd.read_csv(file_ext)

# 轉換合併鍵
if 'LocalTime' in df_wh.columns:
    df_wh['LocalTime'] = pd.to_datetime(df_wh['LocalTime']).dt.date
else:
    raise ValueError('缺少 LocalTime 欄位於 Wh 檔案')

if 'Date' in df_ext.columns:
    # Date 可能已經是 YYYY-MM-DD 格式
    df_ext['Date'] = pd.to_datetime(df_ext['Date']).dt.date
else:
    # 如果沒有 Date 欄，可用 Year/Month/Day 組合
    if all(c in df_ext.columns for c in ('Year','Month','Day')):
        df_ext['Date'] = pd.to_datetime(df_ext[['Year','Month','Day']]).dt.date
    else:
        raise ValueError('外部檔案缺少可用於日期的欄位')

# 檢查鍵範圍
print(f"Wh 檔日期範圍: {df_wh['LocalTime'].min()} ~ {df_wh['LocalTime'].max()}")
print(f"外部檔日期範圍: {df_ext['Date'].min()} ~ {df_ext['Date'].max()}")

# 合併
merged = df_wh.merge(df_ext, left_on='LocalTime', right_on='Date', how='left', suffixes=("","_ext"))

# 移除重複 Date 欄
if 'Date' in merged.columns:
    merged = merged.drop(columns=['Date'])


# 輸出到 output 資料夾，若不存在則建立
output_dir = base_dir.parent / DEFAULT_OUTPUT_SUBDIR
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / DEFAULT_OUTPUT_NAME
merged.to_csv(output_file, index=False)

print(f"合併完成: {len(merged)} 筆資料")
print(f"欄位總數: {len(merged.columns)}")
print(f"輸出檔案: {output_file}")
print(merged.head())
