import pandas as pd
import re
from pathlib import Path

p = Path("input/SolarRecord_260310_1829-row.csv")
df = pd.read_csv(p, low_memory=False)

# 過濾掉標題行
df = df[df['LocalTime'].astype(str).str.strip() != 'LocalTime']

ts_series = df['LocalTime'].astype(str).str.strip().head(50000)

# 嘗試補零
def zero_pad(s):
    import re
    # 規則1：月日無零補位
    s = re.sub(
        r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',
        lambda m: f"{m.group(1)}/{m.group(2).zfill(2)}/{m.group(3).zfill(2)}",
        s
    )
    # 規則2：小時無零補位（帶秒）
    s = re.sub(
        r'\s(\d{1,2}):(\d{2}):(\d{2})',
        lambda m: f" {m.group(1).zfill(2)}:{m.group(2)}:{m.group(3)}",
        s
    )
    # 規則3：小時無零補位（無秒）
    s = re.sub(
        r'\s(\d{1,2}):(\d{2})(?!\d)',
        lambda m: f" {m.group(1).zfill(2)}:{m.group(2)}",
        s
    )
    return s

ts_padded = ts_series.apply(zero_pad)

# 嘗試解析
parsed_orig = pd.to_datetime(ts_series, errors='coerce')
parsed_padded = pd.to_datetime(ts_padded, errors='coerce')

na_orig = (parsed_orig.isna()).sum()
na_padded = (parsed_padded.isna()).sum()

print(f"原始格式 NaT: {na_orig}")
print(f"補零後 NaT: {na_padded}")

# 找出仍然失敗的樣本
still_failed = ts_padded[(parsed_padded.isna()) & (ts_series != 'LocalTime')].head(20)
print(f"\n補零後仍失敗的樣本：")
for idx, val in enumerate(still_failed, 1):
    print(f"  {idx}. '{val}'")
