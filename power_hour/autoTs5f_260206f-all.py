# 區段 load my csv

# Google Drive（掛載）
# from google.colab import drive
# drive.mount('/content/drive')

print("autoTS.py start !!!")

import pandas as pd
# === 1. 讀取資料 ===
csv_path = r'T:\OneDrive\1TB\School\SolarRecord\SolarRecord(260204)_h_fillna_WithCodis.csv'


df = pd.read_csv(csv_path)
# 看一下內容
print("df.head()")
print(df.head())
print("df.tail()")
print(df.tail())

# ===

# === 2. autoTS
## superfast

# 1️⃣ 安裝必要套件
# !pip install autots pandas numpy

import pandas as pd
from autots import AutoTS

# 2️⃣ 載入資料
#csv_path = "/content/drive/MyDrive/Colab_SolarRecord/data/SolarRecordWithCodis202504-12.csv"
#csv_path = "/content/drive/MyDrive/Colab_SolarRecord/data/codis/codis202504-12-fix2.csv"
# csv_path = '/content/drive/MyDrive/Colab_SolarRecord/data/SolarRecord(260204)_h_fillna_WithCodis.csv'
csv_path = r'T:\OneDrive\1TB\School\SolarRecord\SolarRecord(260204)_h_fillna_WithCodis.csv'
# output_path = "/content/drive/MyDrive/Colab_SolarRecord/data/AutoTs4_Superfast.csv"
output_path = r"T:\OneDrive\1TB\School\SolarRecord\AutoTs7_Superfast.csv"
df = pd.read_csv(csv_path)  # 你的 CSV 檔名

# 3️⃣ 將時間欄轉成 datetime，排序，去重
df['LocalTime'] = pd.to_datetime(df['LocalTime'], errors='coerce')
df = df.dropna(subset=['LocalTime'])  # 去掉無效時間
df = df.sort_values('LocalTime').drop_duplicates('LocalTime')

# 🔥 4️⃣ 篩選時間區間（保留 2025-04-23 到 2025-12-01）
start_date = pd.to_datetime("2025-04-23")
end_date   = pd.to_datetime("2026-02-04")
df = df[(df['LocalTime'] >= start_date) & (df['LocalTime'] <= end_date)]

# 4️⃣ 設定 Prophet / AutoTS 格式
df_ts = df[['LocalTime', 'W']].rename(columns={'LocalTime': 'ds', 'W': 'y'})

# 同一天同小時平均補法
# 例如缺 14:00 → 用所有天的 14:00 平均
df_ts['hour'] = df_ts['ds'].dt.hour
df_ts['y'] = df_ts['y'].fillna(df_ts.groupby('hour')['y'].transform('mean'))

# --- 修正開始 ---
# 6️⃣ 建立 AutoTS 模型
# 調整 AutoTS 參數以更好地處理長期預測，並避免 'window shape' 錯誤
Current_forecast_length = 24*15*1
print(f" ### running. model = AutoTS(forecast_length = {Current_forecast_length})")
# my_model_list='superfast'
# my_model_list=[
#     'ETS',
#     'Theta',
#     'SeasonalNaive',
#     'Prophet',
#     'GLM',
#     'AverageValueNaive'
#     ],
# my_model_list='all',
model = AutoTS(
    #forecast_length=24,      # 讓 AutoTS 進行內部交叉驗證的長度（例如，一週=168小時）
    #forecast_length = 24*1*1,
    #forecast_length = 24*7*1,
    forecast_length = Current_forecast_length,
    # mixed_length = True, # XX
    frequency='h',           # 時間頻率：每小時
    # ensemble='simple',       # 使用簡單集成
    # ensemble = "weighted", # 適合長期預測
    #ensemble = "horizontal_weighted", # 適合長期預測
    ensemble=['weighted', 'horizontal'],
    prediction_interval=0.9, # 预测区间的置信水平。默认值是0.9 # 0.8 => MASE2.6
    #max_generations=1,      # 增加世代數，讓 AutoTS 有更多時間找到更適合的模型
    #max_generations=20,
    max_generations=50,
    # model_list='superfast',    # 使用 'default' 模型列表，通常更全面且包含更適合長期預測的模型
    # model_list=[
    #     'ETS',
    #     'Theta',
    #     'SeasonalNaive',
    #     'Prophet',
    #     'GLM',
    #     'AverageValueNaive'
    #     ],
    model_list='all',
    #validation_method='backwards', # 往回驗證
    validation_method='Similarity', # 相似度演算法會自動尋找與用於預測的最新資料最相似的資料段。這是目前最佳的通用選擇，但對雜亂的數據可能較為敏感。
    # transformer_list=['PositiveShift'],   # 確保 y > 0
    transformer_list=[
        'ClipOutliers',
        'Detrend',
        'SeasonalDifference'
        ],         # “fast”、“medium”、“all
    #num_validations=1,  # 預設 2 # 設定為3，確保泛用化
    num_validations=3,  # 預設 2 # 設定為3，確保泛用化
    no_negatives=True, # 當您的資料預期始終為 0 或更大值（例如銷售量）時，一個方便的參數是設定該參數no_negatives=True。這將強制預測值大於或等於 0。
)

# 7️⃣ 擬合模型
model = model.fit(df_ts, date_col='ds', value_col='y')

# 8️⃣ 預測 - 指定預測長度（一年=8760小時）
future_steps = 24 * 7
prediction = model.predict(forecast_length=future_steps)
forecast = prediction.forecast

# 存檔
forecast.to_csv(output_path)
print(f"已儲存到 {output_path}")

print("預測發電量：")
print(forecast.head())

### ===
### """ 計算 MASE、MAE、R²、SMAPE，並顯示前幾小時誤差 """
import numpy as np

### function of 計算 MASE、MAE、R²、SMAPE，並顯示前幾小時誤差
def compute_mase2_and_error(df_ts, forecast, debug_hours=5):
    """
    計算 MASE、MAE、R²、SMAPE，並顯示前幾小時誤差
    """

    # 1️⃣ 原始序列
    y_true = df_ts['y'].values

    # 2️⃣ 預測值
    y_pred = forecast.values.flatten()

    # 3️⃣ 對齊真實值
    y_true_last = y_true[-len(y_pred):]

    # 4️⃣ MAE
    mae = np.mean(np.abs(y_true_last - y_pred))

    # 5️⃣ naive baseline（前一值）
    naive_mae = np.mean(
        np.abs(y_true[-len(y_pred)-1:-1] - y_true_last)
    )

    # 6️⃣ MASE
    mase = mae / naive_mae if naive_mae != 0 else np.inf

    # 7️⃣ R²
    ss_res = np.sum((y_true_last - y_pred) ** 2)
    ss_tot = np.sum((y_true_last - np.mean(y_true_last)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    # 8️⃣ SMAPE
    denominator = (np.abs(y_true_last) + np.abs(y_pred))
    smape = np.mean(
        np.where(denominator == 0, 0, 2 * np.abs(y_true_last - y_pred) / denominator)
    )

    # 印結果
    print(f"\nMAE   : {mae:.4f}")
    print(f"MASE  : {mase:.4f}")
    print(f"R²    : {r2:.4f}")
    print(f"SMAPE : {smape:.4f}\n")

    # 9️⃣ debug：前 N 小時誤差
    error = y_true_last - y_pred
    # print("前幾小時預測誤差：")
    # for i in range(min(debug_hours, len(error))):
    #     print(
    #         f"Hour {i+1}: "
    #         f"True={y_true_last[i]:.2f}, "
    #         f"Pred={y_pred[i]:.2f}, "
    #         f"Error={error[i]:.2f}"
    #     )

    return {
        "MAE": mae,
        "MASE": mase,
        "R2": r2,
        "SMAPE": smape,
        "error": error
    }

###
print(" \n### 預測future_steps： 24 * 1 :")
future_steps = 24 * 1
prediction = model.predict(forecast_length=future_steps)
forecast = prediction.forecast
compute_mase2_and_error(df_ts, forecast)
###
print(" ### 預測future_steps： 24 * 7 :")
future_steps = 24 * 7
prediction = model.predict(forecast_length=future_steps)
forecast = prediction.forecast
compute_mase2_and_error(df_ts, forecast)
###
print(" ### 預測future_steps： 24 * 15 :")
future_steps = 24 * 15
prediction = model.predict(forecast_length=future_steps)
forecast = prediction.forecast
compute_mase2_and_error(df_ts, forecast)
###
print(" ### 預測future_steps： 24 * 30 :")
future_steps = 24 * 30
prediction = model.predict(forecast_length=future_steps)
forecast = prediction.forecast
compute_mase2_and_error(df_ts, forecast)
print(f" ### end. model = AutoTS(forecast_length = {Current_forecast_length})")

####
# 預測發電量：
#                             y
# 2026-02-04 00:00:00  0.005919
# 2026-02-04 01:00:00  0.003100
# 2026-02-04 02:00:00  0.002600
# 2026-02-04 03:00:00  0.002200
# 2026-02-04 04:00:00  0.008300

# ### 預測future_steps： 24 * 1 :

# MAE   : 0.0200
# MASE  : 0.3976
# R²    : 0.9784
# SMAPE : 0.2126

#  ### 預測future_steps： 24 * 7 :

# MAE   : 0.1393
# MASE  : 2.5046
# R²    : 0.2642
# SMAPE : 0.8103

#  ### 預測future_steps： 24 * 15 :

# MAE   : 0.2764
# MASE  : 5.0883
# R²    : -1.0097
# SMAPE : 1.3062

#  ### 預測future_steps： 24 * 30 :

# MAE   : 0.2309
# MASE  : 4.3416
# R²    : -0.7099
# SMAPE : 1.1594