"""autoTs_weather.py

從合併後的氣象 CSV 讀取日資料，並針對數值欄位（每一個氣候變數）
分別使用 AutoTS 預測未來一年。

使用方法：
    python autoTs_weather.py

預測結果會輸出到 workspace 下的 forecasts/ 資料夾，
每一個變數會生成 forecast_<column>.csv。

注意：請先安裝 autots（pip install autots pandas numpy）。

"""

"""
這支 autoTs_weather.py 的功能是：

讀入氣象日資料 CSV
autoTs_weather.py 會讀 csv/2020-01--2026-02.csv，並把 Date 解析成日期、排序、去重。

自動挑選要預測的欄位
它抓所有「數值欄位」，再排除 Year、Month、Day、ObsTime，剩下的每個氣象變數都會各自建模。

對每個變數用 AutoTS 訓練與預測
每個欄位都會轉成 ds/y 格式，建立 AutoTS 模型（model_list="superfast"、ensemble="simple" 等參數），然後做時間序列預測。

輸出每個欄位的預測結果
每個變數會輸出一個 CSV 到 forecasts，檔名格式是 forecast_<欄位名>.csv。

具備基本容錯
如果某欄位全是空值或模型報錯，會印錯誤並跳過，不會讓整個程式中斷。

補充：程式目前 forecast_length = 7，也就是預測 7 天；但註解寫的是「預測未來一年/365 天」，這兩者不一致。
"""

import os
import pandas as pd
import numpy as np
import glob

try:
    from autots import AutoTS
except ImportError:
    raise ImportError("需要安裝 autots：pip install autots")


def calc_smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    valid = denom != 0
    if not np.any(valid):
        return np.nan
    return float(np.mean(2.0 * np.abs(y_pred[valid] - y_true[valid]) / denom[valid]) * 100.0)


def calc_mase(y_true, y_pred, y_train):
    """Mean Absolute Scaled Error (naive lag-1 baseline)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) < 2:
        return np.nan
    naive_scale = np.mean(np.abs(np.diff(y_train)))
    if naive_scale == 0 or np.isnan(naive_scale):
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / naive_scale)


def calc_r2(y_true, y_pred):
    """R-squared."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1 - (ss_res / ss_tot))

# 路徑設定
# original assumed csv folder under script; adjust to workspace root
base_dir = os.path.dirname(os.path.dirname(__file__))  # parent of autoTs_weather
# you can change filename as needed
csv_filename = "202503--202602-d.csv"
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
    "num_validations": 1,
}

metrics_rows = []

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

    # 使用最後 forecast_length 天作為驗證集，計算 MASE/SMAPE/R2
    if len(ts) > forecast_length + 1:
        train_ts = ts.iloc[:-forecast_length].copy()
        valid_ts = ts.iloc[-forecast_length:].copy()

        try:
            valid_model = AutoTS(**model_kwargs)
            valid_model = valid_model.fit(train_ts, date_col="ds", value_col="y")
            valid_forecast_df = valid_model.predict(forecast_length=forecast_length).forecast

            pred_series = valid_forecast_df.iloc[:, 0].copy()
            pred_series.index = pd.to_datetime(pred_series.index)
            actual_series = valid_ts.set_index("ds")["y"].copy()

            compare = pd.DataFrame({"actual": actual_series, "pred": pred_series}).dropna()
            if compare.empty:
                print(f"  (欄位 {col} 驗證預測無法對齊，略過指標計算)")
            else:
                y_true = compare["actual"].values
                y_pred = compare["pred"].values
                y_train = train_ts["y"].values
                mase = calc_mase(y_true, y_pred, y_train)
                smape = calc_smape(y_true, y_pred)
                r2 = calc_r2(y_true, y_pred)
                metrics_rows.append(
                    {
                        "column": col,
                        "n_train": int(len(train_ts)),
                        "n_valid": int(len(compare)),
                        "MASE": mase,
                        "SMAPE": smape,
                        "R2": r2,
                    }
                )
                print(f"  驗證指標 MASE={mase:.4f}, SMAPE={smape:.2f}%, R2={r2:.4f}")
        except Exception as exc:
            print(f"  針對 {col} 計算驗證指標發生錯誤: {exc}")
    else:
        print(f"  (欄位 {col} 資料不足，無法計算 MASE/SMAPE/R2)")

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

if metrics_rows:
    metrics_df = pd.DataFrame(metrics_rows).sort_values("column")
    metrics_path = os.path.join(output_dir, "forecast_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"已儲存模型指標到 {metrics_path}")
else:
    print("未產生任何模型指標。")

# === 合併所有預測檔案 ===
print("\n開始合併所有預測檔案")
forecast_files = [
    f
    for f in glob.glob(os.path.join(output_dir, "forecast_*.csv"))
    if os.path.basename(f) != "forecast_metrics.csv"
]
if forecast_files:
    merged = None
    for f in forecast_files:
        df_f = pd.read_csv(f)
        if "Date" not in df_f.columns:
            if "index" in df_f.columns:
                df_f = df_f.rename(columns={"index": "Date"})
            elif "ds" in df_f.columns:
                df_f = df_f.rename(columns={"ds": "Date"})
            elif "Unnamed: 0" in df_f.columns:
                df_f = df_f.rename(columns={"Unnamed: 0": "Date"})

        if "Date" not in df_f.columns:
            print(f"略過 {os.path.basename(f)}: 找不到 Date 欄位")
            continue

        df_f["Date"] = pd.to_datetime(df_f["Date"], errors="coerce")
        df_f = df_f.dropna(subset=["Date"])
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
