"""
將 SolarRecord(260204)_h_fillna.csv 降頻率為每天，Wh 欄位使用加總，其它欄位使用平均
"""
import pandas as pd
import numpy as np
from pathlib import Path

# 可修改的輸入/輸出路徑（放在檔案頂端方便維護）
# 修改下面兩行即可變更輸入與輸出檔案
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / 'SolarRecord_260310_1829-hour-Wh.csv'
OUTPUT_FILE = SCRIPT_DIR / 'SolarRecord_260310_1829-daily.csv'

# 讀取輸入檔案
df = pd.read_csv(INPUT_FILE)

# 清理完全空白的欄位
df = df.dropna(axis=1, how='all')

print(f"✓ 讀取檔案: {INPUT_FILE}")
print(f"  原始記錄數: {len(df)}")

# 轉換時間欄位
if 'LocalTime' in df.columns:
    df['LocalTime'] = pd.to_datetime(df['LocalTime'])
else:
    raise ValueError('找不到 LocalTime 欄位。')

# 新增日期分組欄
df['Date'] = df['LocalTime'].dt.date

# 移除不需要的欄位（若存在）
df = df.drop(columns=['hour'], errors='ignore')

# 定義聚合規則
agg_dict = {}
for col in df.columns:
    if col in ('LocalTime', 'Date'):
        continue
    if col == 'Wh':
        agg_dict[col] = 'sum'
    elif pd.api.types.is_numeric_dtype(df[col]):
        agg_dict[col] = 'mean'

# 執行每天聚合
daily = df.groupby('Date').agg(agg_dict).reset_index()
# 轉回 LocalTime
daily['LocalTime'] = pd.to_datetime(daily['Date'])
daily = daily[['LocalTime'] + [c for c in daily.columns if c not in ('LocalTime', 'Date')]]

# 排序
daily = daily.sort_values('LocalTime').reset_index(drop=True)

# 確保輸出目錄存在，並寫出檔案
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
daily.to_csv(OUTPUT_FILE, index=False)

print(f"\n✓ 處理完成！")
print(f"  輸入檔案: {INPUT_FILE}")
print(f"  輸出檔案: {OUTPUT_FILE}")
print(f"  原始記錄數: {len(df)}")
print(f"  每日記錄數: {len(daily)}")
print(daily.head())