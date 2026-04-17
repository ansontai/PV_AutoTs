import pandas as pd
import glob
import os

# 設定資料夾路徑
folder_path = r"T:\OneDrive\1TB\School\台灣氣象局資料庫\台中-北區精武路295號\day_unit"   # 如果在本機就改成你的資料夾路徑

# 抓所有類似檔名
file_list = glob.glob(os.path.join(folder_path, "467490-*.csv"))

all_data = []

for file in file_list:
    print("讀取:", file)
    
    # 讀檔 (第一行是中文欄名，第二行才是英文字頭)
    # 嘗試跳過第一行，使欄名為英文
    df = pd.read_csv(file, header=1)
    
    # 從檔名解析 年月，容許 Windows 會加上複製標記 like " (1)"
    import re
    filename = os.path.basename(file)
    # 例如 467490-2025-12.csv  或 467490-2001-11 (1).csv
    parts = filename.replace(".csv", "").split("-")
    year = int(parts[1])
    # 清理月份部分，只保留數字
    month_str = re.search(r"\d+", parts[2])
    if month_str:
        month = int(month_str.group())
    else:
        raise ValueError(f"無法從檔名取出月份: {filename}")
    
    # 如果你有 yyyymmddhh 這種欄位
    if "yyyymmddhh" in df.columns:
        df["yyyymmddhh"] = df["yyyymmddhh"].astype(str)
        df["Year"] = df["yyyymmddhh"].str[0:4].astype(int)
        df["Month"] = df["yyyymmddhh"].str[4:6].astype(int)
        df["Day"] = df["yyyymmddhh"].str[6:8].astype(int)
    else:
        # 如果沒有時間欄位，就用檔名給的年月
        df["Year"] = year
        df["Month"] = month

    # 建立日期欄 (日資料存在 ObsTime)
    if "ObsTime" in df.columns:
        # 保證兩位數格式，缺失值防護
        df["ObsTime"] = df["ObsTime"].astype(str).str.zfill(2)
        df["Day"] = pd.to_numeric(df["ObsTime"], errors="coerce")
    elif "Day" in df.columns:
        # 可能已經從 yyyymmddhh 拆出來了
        df["Day"] = df["Day"].astype(int)
    else:
        # 沒有日欄位的話預設為 1
        df["Day"] = 1

    df["Date"] = pd.to_datetime(
        df["Year"].astype(int).astype(str) + "-" +
        df["Month"].astype(int).astype(str) + "-" +
        df["Day"].astype(int).astype(str), errors="coerce")
    
    all_data.append(df)

# 合併全部
merged_df = pd.concat(all_data, ignore_index=True)

# 輸出
output_path = os.path.join(folder_path, "merged_all.csv")
merged_df.to_csv(output_path, index=False)

print("合併完成！")
print("總筆數:", len(merged_df))