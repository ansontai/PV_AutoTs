# 需要 pandas：pip install pandas
import pandas as pd

path = "PVGIS/raw/tmy_24.148_120.703_2005_2023.csv"

# 找到 header 行（以 "time(UTC)" 開頭）
with open(path, "r", encoding="utf-8") as f:
    header_idx = 0
    for i, line in enumerate(f):
        if line.startswith("time(UTC)"):
            header_idx = i
            break

# 以該行為欄位標題讀入資料，將空字串視為 NA
df = pd.read_csv(path, header=0, skiprows=header_idx, na_values=[""])

# 計算
n_rows = len(df)
n_cells = df.size
n_missing = df.isna().sum().sum()
missing_ratio = n_missing / n_cells if n_cells else 0

print("總筆數（列數）:", n_rows)
print(f"整體缺測: {n_missing} / {n_cells} ＝ {missing_ratio:.4%}")