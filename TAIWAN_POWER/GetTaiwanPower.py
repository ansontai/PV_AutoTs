
import os
import requests
import pandas as pd
from io import StringIO

# 台電每日太陽光電資料 API（政府資料開放平台）
BASE_URL = "https://quality.data.gov.tw/dq_download_csv.php?nid=14261&md5_url="

# 建議你把 md5_url 清單整理成 dict（每個月一個 md5）
# 這裡示範用你自己的 md5 清單
md5_list = {
    "2024-01": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "2024-02": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    # ...
}

def download_csv(md5, save_path):
    url = BASE_URL + md5
    r = requests.get(url)
    r.encoding = "utf-8"

    if r.status_code == 200:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(r.text)
        print(f"✔ 已下載：{save_path}")
    else:
        print(f"✘ 下載失敗：{save_path}")

def merge_all_csv(folder="solar_daily"):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    df_list = []

    for f in files:
        path = os.path.join(folder, f)
        df = pd.read_csv(path)
        df_list.append(df)

    merged = pd.concat(df_list, ignore_index=True)

    # 統一欄位名稱
    merged.columns = (
        merged.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("（", "_")
        .str.replace("）", "")
    )

    return merged

# 主程式
os.makedirs("solar_daily", exist_ok=True)

for ym, md5 in md5_list.items():
    save_path = f"solar_daily/{ym}.csv"
    download_csv(md5, save_path)

merged_df = merge_all_csv()
merged_df.to_csv("solar_daily_merged.csv", index=False, encoding="utf-8-sig")

print("🎉 合併完成！輸出：solar_daily_merged.csv")
