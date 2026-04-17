import pandas as pd
from pathlib import Path

# 讀取檔案
input_file = Path('SolarRecord(260204).csv')
df = pd.read_csv(input_file)

# 清理空白欄位
df = df.dropna(axis=1, how='all')

# 轉換時間欄位為 datetime
df['LocalTime'] = pd.to_datetime(df['LocalTime'])

# 提取日期（用於分組）
df['Date'] = df['LocalTime'].dt.date

# 定義聚合規則
agg_dict = {}
for col in df.columns:
    if col == 'LocalTime' or col == 'Date':
        continue
    elif col == 'W':
        # W 欄位使用 sum
        agg_dict[col] = 'sum'
    elif pd.api.types.is_numeric_dtype(df[col]):
        # 只對數值欄位使用 mean
        agg_dict[col] = 'mean'
    # 非數值欄位直接跳過

# 按日期分組並聚合
daily_df = df.groupby('Date').agg(agg_dict).reset_index()

# 將日期轉回為時間
daily_df['LocalTime'] = pd.to_datetime(daily_df['Date'])
daily_df = daily_df[['LocalTime'] + [col for col in daily_df.columns if col != 'LocalTime' and col != 'Date']]

# 排序
daily_df = daily_df.sort_values('LocalTime').reset_index(drop=True)

# 輸出檔案
output_file = 'SolarRecord(260204)_daily.csv'
daily_df.to_csv(output_file, index=False)

print(f"✓ 處理完成！")
print(f"  輸入檔案: {input_file}")
print(f"  輸出檔案: {output_file}")
print(f"  原始記錄數: {len(df)}")
print(f"  每日記錄數: {len(daily_df)}")
print(f"\n前5筆每日資料：")
print(daily_df.head())
