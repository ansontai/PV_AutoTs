import pandas as pd
from pathlib import Path

p = Path("input/SolarRecord_260310_1829-row.csv")
df = pd.read_csv(p, low_memory=False)

# 過濾標題行，重置索引（與正式腳本一致）
df = df[df['LocalTime'].astype(str).str.strip() != 'LocalTime'].reset_index(drop=True)

# 獲取前幾個時間值
sample = df['LocalTime'].head(280000).tail(100)

print("檢查樣本時間字符串（bytes 層級）:")
for idx, val in enumerate(sample.head(10)):
    val_str = str(val).strip()
    val_bytes = val_str.encode('utf-8')
    print(f"\nIndex {df.index[idx if idx < len(df) else -1]}:")
    print(f"  Value:     '{val_str}'")
    print(f"  Bytes:     {val_bytes}")
    print(f"  Hex:       {val_bytes.hex()}")
    print(f"  Length:    {len(val_str)}")
    
    # 試試解析
    parsed = pd.to_datetime(val_str, errors='coerce')
    print(f"  Parsed:    {parsed}")
