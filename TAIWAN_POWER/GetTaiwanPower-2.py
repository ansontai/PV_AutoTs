import os
import sys
import requests
import pandas as pd
from io import StringIO

# 政府資料開放平台 API（台電太陽光電每日發電量）
DATASET_API = "https://data.gov.tw/api/v1/rest/dataset/14261"

# 輸出路徑（以目前執行檔案為基準）
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "output")
solar_dir = os.path.join(output_dir, "solar_daily")
os.makedirs(solar_dir, exist_ok=True)

# 取得所有下載連結（每個月一個 CSV）
print(f"Fetching dataset API: {DATASET_API}")
try:
    resp = requests.get(DATASET_API, timeout=30)
except Exception as e:
    print("✘ 請求 API 失敗：", e)
    sys.exit(1)

if resp.status_code != 200:
    print(f"✘ API 回傳狀態：{resp.status_code}")
    print(resp.text[:1000])
    sys.exit(1)

try:
    data = resp.json()
except Exception as e:
    print("✘ 解析 API JSON 失敗：", e)
    print(resp.text[:1000])
    sys.exit(1)

resources = data.get("result", {}).get("resources", [])

csv_links = [
    r["download_url"]
    for r in resources
    if r["format"].lower() == "csv"
]

print(f"找到 {len(csv_links)} 個 CSV 檔案，開始下載…")

if not csv_links:
    print("沒有可下載的 CSV，程序結束。")
    sys.exit(0)

# 下載所有 CSV
for url in csv_links:
    filename = url.split("/")[-1]
    save_path = os.path.join(solar_dir, filename)

    try:
        r = requests.get(url, timeout=60)
        r.encoding = "utf-8"
        if r.status_code == 200:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(r.text)
            print(f"✔ 已下載：{filename}")
        else:
            print(f"✘ 下載失敗（狀態碼 {r.status_code}）：{filename}")
    except Exception as e:
        print(f"✘ 下載例外：{filename} -> {e}")

# 合併所有 CSV
files = [f for f in os.listdir(solar_dir) if f.endswith(".csv")]
df_list = []

for f in files:
    path = os.path.join(solar_dir, f)
    try:
        df = pd.read_csv(path)
        df_list.append(df)
    except Exception as e:
        print(f"跳過無法讀取的檔案：{f} -> {e}")

if not df_list:
    print("沒有可合併的 CSV 檔案，程序結束。")
    sys.exit(0)

merged = pd.concat(df_list, ignore_index=True)

# 清洗欄位名稱
merged.columns = (
    merged.columns.str.strip()
    .str.replace(" ", "_")
    .str.replace("（", "_")
    .str.replace("）", "")
)

# 輸出合併檔案
merged_path = os.path.join(output_dir, "solar_daily_merged.csv")
merged.to_csv(merged_path, index=False, encoding="utf-8-sig")

print(f"🎉 合併完成！輸出：{merged_path}")
