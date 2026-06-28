import pandas as pd
from pathlib import Path

p = Path("input/SolarRecord_260310_1829-row.csv")
df = pd.read_csv(p, low_memory=False)

# 直接查看指定索引的 LocalTime 值
failed_indices = [231097, 277719, 277720, 277721, 277722]
print("直接查看失敗索引的 LocalTime 值：")
for idx in failed_indices:
    if idx < len(df):
        val = df.iloc[idx]['LocalTime']
        print(f"  Index {idx}: '{val}' (type: {type(val).__name__})")

# 掃描整個 LocalTime 欄位，找出所有無法被 pd.to_datetime 解析的值
print("\n掃描全部失敗的值...")
ts = df['LocalTime'].astype(str).str.strip()
ts = ts[ts != 'LocalTime']
parsed = pd.to_datetime(ts, errors='coerce')
failed_mask = parsed.isna() & ts.notna()

if failed_mask.any():
    unique_failed = ts[failed_mask].unique()
    print(f"總共有 {len(unique_failed)} 種失敗的值（前20種）:")
    for i, val in enumerate(unique_failed[:20], 1):
        print(f"  {i}. '{val}'")
