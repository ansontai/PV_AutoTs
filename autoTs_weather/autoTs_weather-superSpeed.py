"""autoTs_weather.py

從合併後的氣象 CSV 讀取日資料，並針對數值欄位（每一個氣候變數）
分別使用 AutoTS 預測未來一年。

使用方法：
    python autoTs_weather.py

預測結果會輸出到 workspace 下的 forecasts/ 資料夾，
每一個變數會生成 forecast_<column>.csv。

注意：請先安裝 autots（pip install autots pandas numpy）。

"""

import os
import pandas as pd
import numpy as np
import glob

try:
    from autots import AutoTS
except ImportError:
    raise ImportError("需要安裝 autots：pip install autots")

# 路徑設定
# original assumed csv folder under script; adjust to workspace root
base_dir = os.path.dirname(os.path.dirname(__file__))  # parent of autoTs_weather
# you can change filename as needed
csv_filename = "202102--202602-d.csv"
csv_path = os.path.join(base_dir, "csv", csv_filename)
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"預期的 CSV 檔案不存在：{csv_path}")
output_dir = os.path.join(os.path.dirname(__file__), "forecasts")
os.makedirs(output_dir, exist_ok=True)

# 讀取資料
print(f"讀取資料: {csv_path}")
df = pd.read_csv(csv_path, parse_dates=["Date"], dayfirst=False)
df = df.sort_values("Date").drop_duplicates("Date")

# 將 Date 設為 index 方便後續
#df = df.set_index('Date')  # 不一定需要

# 選擇要預測的欄位：
# 先排除日期索引和明顯不應該預測的欄位，之後在迴圈中轉換為數值型
exclude = set(["Date", "Year", "Month", "Day", "ObsTime"])
target_cols = [c for c in df.columns if c not in exclude]

print("將針對以下欄位逐一訓練與預測:")
print(target_cols)

# 預測參數
forecast_length = 3  # 預測未來 365 天
frequency = "D"

# 如果需要調整 AutoTS 參數，可在這裡修改 model_kwargs
model_kwargs = {
    "forecast_length": forecast_length,
    "frequency": frequency,
    # 我們使用較簡單的 ensemble 以加快速度，必要時可改成 ['weighted','horizontal'] 等
    "ensemble": "simple",
    "model_list": "superfast",  # 或 'all' 依需求
    "prediction_interval": 0.9,
    # 基本的 transformer 參數
    "transformer_list": ["ClipOutliers", "Detrend", "SeasonalDifference"],
    # 減少訓練時間
    "max_generations": 1,
    "num_validations": 1,
}

for col in target_cols:
    print(f"\n---- column: {col} ----")
    ts = df[["Date", col]].copy()
    # 強制轉成數值，非數值將變為 NaN
    ts[col] = pd.to_numeric(ts[col], errors="coerce")
    ts = ts.dropna()
    if ts.empty:
        print(f"  (欄位 {col} 無任何可用數值資料，跳過)")
        continue
    ts = ts.rename(columns={"Date": "ds", col: "y"})

    try:
        model = AutoTS(**model_kwargs)
        model = model.fit(ts, date_col="ds", value_col="y")
        prediction = model.predict(forecast_length=forecast_length)
        forecast = prediction.forecast
    except Exception as exc:
        print(f"  針對 {col} 執行 AutoTS 發生錯誤: {exc}")
        continue

    # 加上日期欄存檔
    forecast = forecast.reset_index()
    forecast.rename(columns={"index": "Date"}, inplace=True)
    outpath = os.path.join(output_dir, f"forecast_{col}.csv")
    forecast.to_csv(outpath, index=False)
    print(f"  儲存預測結果到 {outpath}")

print("\n所有變數預測完成。")

# === 合併所有預測檔案 ===
print("\n開始合併所有預測檔案")
forecast_files = glob.glob(os.path.join(output_dir, "forecast_*.csv"))
if forecast_files:
    merged = None
    for f in forecast_files:
        df_f = pd.read_csv(f, parse_dates=["Date"])
        # 改列名 y -> 視變數名稱
        colname = os.path.splitext(os.path.basename(f))[0].replace("forecast_", "")
        if "y" in df_f.columns:
            df_f = df_f.rename(columns={"y": colname})
        else:
            # 若先前已改過索引欄名
            other_cols = [c for c in df_f.columns if c != "Date"]
            if other_cols:
                df_f = df_f.rename(columns={other_cols[0]: colname})
        if merged is None:
            merged = df_f
        else:
            merged = pd.merge(merged, df_f, on="Date", how="outer")
    if merged is not None:
        merged = merged.sort_values("Date")
        merged_path = os.path.join(output_dir, "merged_forecasts.csv")
        merged.to_csv(merged_path, index=False)
        print(f"已儲存合併預測到 {merged_path}")
else:
    print("未找到任何預測檔案。")
