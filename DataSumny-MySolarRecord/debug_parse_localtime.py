import pandas as pd
from pathlib import Path

path = Path(r'input/SolarRecord_260310_1829-row.csv')
df = pd.read_csv(path, low_memory=False)

print("=== LocalTime 解析診斷 ===")
print(f"總筆數: {len(df)}")

# 原始解析
parsed = pd.to_datetime(df['LocalTime'], errors='coerce')
mask_na = parsed.isna() & df['LocalTime'].notna()
na_count = mask_na.sum()
print(f"NaT 筆數（原始 pd.to_datetime）: {na_count}")

if na_count > 0:
    print("\n失敗樣本（前20筆）:")
    failed_samples = df.loc[mask_na, 'LocalTime'].head(20)
    for i, val in enumerate(failed_samples, 1):
        print(f"  {i}. '{val}' (type: {type(val).__name__})")

# 嘗試其他解析方式
print("\n=== 嘗試其他解析方式 ===")

# 方式1: infer_datetime_format
parsed_infer = pd.to_datetime(df['LocalTime'], errors='coerce', infer_datetime_format=True)
mask_infer_na = parsed_infer.isna() & df['LocalTime'].notna()
print(f"infer_datetime_format=True: NaT {mask_infer_na.sum()} 筆")

# 方式2: format 明確指定
try:
    parsed_fmt = pd.to_datetime(df['LocalTime'], format='%Y/%m/%d %H:%M', errors='coerce')
    mask_fmt_na = parsed_fmt.isna() & df['LocalTime'].notna()
    print(f"format='%Y/%m/%d %H:%M': NaT {mask_fmt_na.sum()} 筆")
except Exception as e:
    print(f"format 解析失敗: {e}")

# 方式3: 多格式 fallback
def parse_multi_format(ts_series):
    formats = ['%Y/%m/%d %H:%M', '%Y/%m/%d %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M']
    result = pd.Series([pd.NaT] * len(ts_series), index=ts_series.index)
    mask = ts_series.notna()
    for fmt in formats:
        still_na = result[mask].isna()
        if not still_na.any():
            break
        try:
            parsed = pd.to_datetime(ts_series[mask][still_na], format=fmt, errors='coerce')
            result.loc[mask][still_na] = parsed
        except:
            pass
    return result

parsed_multi = parse_multi_format(df['LocalTime'])
mask_multi_na = parsed_multi.isna() & df['LocalTime'].notna()
print(f"多格式 fallback: NaT {mask_multi_na.sum()} 筆")

if mask_multi_na.any():
    print("\n多格式後仍失敗的樣本（前10筆）:")
    still_failed = df.loc[mask_multi_na, 'LocalTime'].head(10)
    for i, val in enumerate(still_failed, 1):
        print(f"  {i}. '{val}'")
