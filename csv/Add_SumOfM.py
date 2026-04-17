import pandas as pd

# 讀入（若路徑不同，請調整）
df = pd.read_csv("csv/SolarRecord_260310_Wh-hour.csv", parse_dates=["LocalTime"])

# 設時間索引
df = df.set_index("LocalTime")

# 每日 Wh（把每小時 Wh 加總為每天）
daily_wh = df["Wh"].resample("D").sum()

# 每月 Wh（把每日加總再依月加總）
# pandas 新版已不再支援 'M' offsets，改用 'ME' (month end) 或 'MS' (month start)
monthly_wh = daily_wh.resample("ME").sum()

# 轉成較易讀的月份欄位（YYYY-MM）
out = monthly_wh.rename("Wh_sum").to_frame()
out.index = out.index.to_period("M").astype(str)

# 寫出結果
out.to_csv("csv/SolarRecord_260310_Wh-monthly-sum.csv")
print("輸出檔案：csv/SolarRecord_260310_Wh-monthly-sum.csv")
print(out)