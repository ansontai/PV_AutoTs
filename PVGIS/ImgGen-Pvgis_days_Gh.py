import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 讀取資料（自動定位 time(UTC) 標頭列，避免 skiprows 固定值失敗）
csv_path = Path(__file__).resolve().parent / "raw" / "tmy_24.148_120.703_2005_2023.csv"
header_row = None
with open(csv_path, "r", encoding="utf-8") as f:
	for i, line in enumerate(f):
		if line.startswith("time(UTC)"):
			header_row = i
			break

if header_row is None:
	raise ValueError(f"Cannot find 'time(UTC)' header in: {csv_path}")

df = pd.read_csv(csv_path, skiprows=header_row)

# 選一個年份（例如 2018）
df['time(UTC)'] = df['time(UTC)'].astype(str)
df_year = df[df['time(UTC)'].str.startswith('2018')].copy()

# 取日期（YYYYMMDD）
df_year['date'] = df_year['time(UTC)'].str[:8]

# 確保 GH 數值格式正確
df_year['G(h)'] = pd.to_numeric(df_year['G(h)'], errors='coerce')

# ✅ 每日總量（用 sum，不是平均）
daily_gh = df_year.groupby('date')['G(h)'].sum()

# 畫圖
plt.figure(figsize=(16,5))
plt.plot(daily_gh.index, daily_gh.values)

plt.title('2018 GH')
plt.xlabel('Date')
plt.ylabel('GH (Wh/m²)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()