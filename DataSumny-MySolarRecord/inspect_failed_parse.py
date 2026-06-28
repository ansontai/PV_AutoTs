import pandas as pd
from pathlib import Path

p = Path("input/SolarRecord_260310_1829-row.csv")
df = pd.read_csv(p, low_memory=False)

# 用目前的解析方式嘗試
formats = [
    '%Y/%m/%d %H:%M',
    '%Y/%m/%d %H:%M:%S',
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%d %H:%M',
]

result = pd.Series([pd.NaT] * len(df), index=df.index, dtype='datetime64[ns]')
remaining_mask = pd.Series([True] * len(df), index=df.index)

for fmt in formats:
    if not remaining_mask.any():
        break
    try:
        parsed = pd.to_datetime(df.loc[remaining_mask, 'LocalTime'], format=fmt, errors='coerce')
        success_mask = parsed.notna()
        result.loc[remaining_mask][success_mask] = parsed[success_mask]
        remaining_mask = remaining_mask & result.isna()
    except:
        pass

# 最後 fallback
if remaining_mask.any():
    try:
        parsed = pd.to_datetime(df.loc[remaining_mask, 'LocalTime'], errors='coerce', infer_datetime_format=True)
        result.loc[remaining_mask] = parsed
    except:
        pass

# 找失敗的
failed_mask = result.isna() & df['LocalTime'].notna()
failed_count = failed_mask.sum()
print(f"解析失敗: {failed_count} / {len(df)}")

if failed_count > 0:
    print("\n失敗樣本（前30個）:")
    failed_samples = df.loc[failed_mask, 'LocalTime'].head(30)
    for idx, val in enumerate(failed_samples, 1):
        print(f"  {idx}. '{val}'")
