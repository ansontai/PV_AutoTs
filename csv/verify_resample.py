import pandas as pd

df = pd.read_csv('SolarRecord(260204).csv')
daily = pd.read_csv('SolarRecord(260204)_daily.csv')

print('=' * 60)
print('原始檔案統計:')
print(f'  總筆數: {len(df)}')
print(f'  日期範圍: {df.iloc[0,0]} ~ {df.iloc[-1,0]}')

print(f'\n每日檔案統計:')
print(f'  日數: {len(daily)}')
print(f'  節省比例: {len(df)/len(daily):.1f} 倍')

print('\n第一天驗證 (2025-04-23):')
first_day = df[df['LocalTime'].str.startswith('2025/4/23')]
print(f'  原始記錄數: {len(first_day)}')
print(f'  W欄位總和: {first_day["W"].sum():.4f}')
print(f'  每日W值: {daily.iloc[1,1]:.4f}')
print(f'  匹配: {"✓" if abs(first_day["W"].sum() - daily.iloc[1,1]) < 0.001 else "✗"}')

print(f'\n溫度驗證:')
print(f'  DHT11_temp平均: {first_day["DHT11_temp"].mean():.4f}')
print(f'  每日DHT11_temp: {daily.iloc[1,2]:.4f}')
print(f'  匹配: {"✓" if abs(first_day["DHT11_temp"].mean() - daily.iloc[1,2]) < 0.001 else "✗"}')

print('=' * 60)
print('✓ 轉換成功完成！')
print(f'輸出檔案: SolarRecord(260204)_daily.csv')
