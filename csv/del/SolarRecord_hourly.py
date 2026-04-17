"""
將 SolarRecord(260204).csv 降頻率為每小時，缺失的小時使用同一小時區間其他日期的平均值填補
"""
import pandas as pd
import numpy as np
from pathlib import Path

# 讀取原始資料
input_file = Path('SolarRecord(260204).csv')
df = pd.read_csv(input_file)

# 清理完全空白的欄位
df = df.dropna(axis=1, how='all')

print(f"✓ 讀取檔案: {input_file}")
print(f"  原始記錄數: {len(df)}")
print(f"  時間範圍: {df['LocalTime'].min()} 至 {df['LocalTime'].max()}")

# 轉換時間欄位為 datetime
df['LocalTime'] = pd.to_datetime(df['LocalTime'])

# 按小時取整（向下取整）
df['Hour'] = df['LocalTime'].dt.floor('H')

# 分離數值欄位
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# 按小時聚合（mean為主，W欄位特殊處理）
agg_dict = {}
for col in numeric_cols:
    agg_dict[col] = 'mean'

# W欄位如果存在，使用 mean（計算平均功率）
if 'W' in agg_dict:
    agg_dict['W'] = 'mean'

hourly_df = df.groupby('Hour').agg(agg_dict).reset_index()
hourly_df.rename(columns={'Hour': 'LocalTime'}, inplace=True)

print(f"\n聚合後: {len(hourly_df)} 筆小時資料")

# 建立完整的小時時間序列（無缺失）
time_range = pd.date_range(
    start=hourly_df['LocalTime'].min(),
    end=hourly_df['LocalTime'].max(),
    freq='H'
)
full_hours = pd.DataFrame({'LocalTime': time_range})

# 合併，識別缺失的小時
merged_df = full_hours.merge(hourly_df, on='LocalTime', how='left')

# 識別缺失行
missing_mask = merged_df[numeric_cols].isna().any(axis=1)
n_missing = missing_mask.sum()

print(f"缺失的小時數: {n_missing}")

# 對於缺失的小時，使用同小時其他日期的平均值填補
if n_missing > 0:
    # 先提取小時(час)
    merged_df['Hour'] = merged_df['LocalTime'].dt.hour
    
    # 計算每個小時的平均值（所有日期）
    hourly_means = merged_df.groupby('Hour')[numeric_cols].transform('mean')
    
    # 填補缺失值
    for col in numeric_cols:
        mask = merged_df[col].isna()
        merged_df.loc[mask, col] = hourly_means.loc[mask, col]
    
    # 移除臨時欄位
    merged_df = merged_df.drop('Hour', axis=1)
    
    print(f"✓ 已使用同小時平均值填補 {n_missing} 個缺失小時")

# 排序
merged_df = merged_df.sort_values('LocalTime').reset_index(drop=True)

# 輸出檔案
output_file = 'SolarRecord(260204)_hourly_fillna.csv'
merged_df.to_csv(output_file, index=False)

print(f"\n✓ 處理完成！")
print(f"  輸出檔案: {output_file}")
print(f"  小時記錄數: {len(merged_df)}")
print(f"\n前10筆小時資料：")
print(merged_df.head(10))
print(f"\n後5筆小時資料：")
print(merged_df.tail(5))
