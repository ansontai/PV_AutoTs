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

try:
    from autots import AutoTS
except ImportError:
    raise ImportError("需要安裝 autots：pip install autots")

# 路徑設定
csv_path = os.path.join(os.path.dirname(__file__), "csv", "2020-01--2026-02.csv")
output_dir = os.path.join(os.path.dirname(__file__), "forecasts")
os.makedirs(output_dir, exist_ok=True)

# 讀取資料
print(f"讀取資料: {csv_path}")
df = pd.read_csv(csv_path, parse_dates=["Date"], dayfirst=False)
df = df.sort_values("Date").drop_duplicates("Date")

# 將 Date 設為 index 方便後續
#df = df.set_index('Date')  # 不一定需要

# 選擇要預測的欄位：所有數值型欄位（排除年/月/日等）
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# 排除不需要的欄位
exclude = set(["Year", "Month", "Day", "ObsTime"])
target_cols = [c for c in numeric_cols if c not in exclude]

print("將針對以下欄位逐一訓練與預測:")
print(target_cols)

# 預測參數
forecast_length = 7  # 預測未來 365 天
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
    "num_validations": 2,
}

for col in target_cols:
    print(f"\n---- column: {col} ----")
    ts = df[["Date", col]].dropna()
    if ts.empty:
        print("  (沒有可用資料，跳過)")
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
