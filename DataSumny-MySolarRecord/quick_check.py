import pandas as pd
from pathlib import Path

p = Path("input/SolarRecord_260310_1829-row.csv")
df = pd.read_csv(p, low_memory=False)
parsed = pd.to_datetime(df["LocalTime"], errors="coerce")
na_count = (parsed.isna() & df["LocalTime"].notna()).sum()
print(f"NA count: {na_count} / {len(df)}")
print(f"Valid: {(~parsed.isna()).sum()}")
