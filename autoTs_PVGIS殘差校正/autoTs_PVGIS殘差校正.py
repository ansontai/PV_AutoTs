"""
autoTs_PVGIS殘差校正.py

這個程式的目標是對 PVGIS 基準預測做殘差校正，讓最終預測更接近實際觀測值。

整體流程：
1) 讀取 PVGIS 模擬資料與觀測 SolarRecord 資料
2) 把 PVGIS 依目標日期對齊到觀測資料（若日期無法完全對應，則使用日序平均或月平均補值）
3) 計算每月縮放係數 k，使 PVGIS 模擬值在月尺度上與觀測量級匹配
4) 計算觀測值與調整後 PVGIS 之間的殘差
5) 對殘差建立時間序列模型（AutoTS）並預測未來殘差
6) 將未來 PVGIS 預測值乘上比例 k，再加上未來殘差，得到最終校正預測

此架構的核心概念：先保留物理模型（PVGIS）作為基準，再用資料驅動模型修正系統性偏差。
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import json


def read_pvgis(pvgis_fp: Path) -> pd.DataFrame:
	pvgis_fp = Path(pvgis_fp)
	"""
	讀取 PVGIS CSV 並回傳整理過的 DataFrame。

	參數:
	- pvgis_fp: PVGIS CSV 的檔案路徑 (Path 或 字串)，要求有日期欄位 `date`。

	回傳:
	- pd.DataFrame: 含 `date` (datetime, normalize 到日) 與 `P_mapped_Wh` 欄位。

	實作細節:
	- 會檢查檔案是否存在，並使用 `parse_dates` 將 `date` 轉成 datetime。
	- `normalize()` 會把時間部分歸零，確保以日為單位對齊。
	"""
	if not pvgis_fp.exists():
		raise FileNotFoundError(f"找不到 PVGIS 檔案：{pvgis_fp}")
	pvgis = pd.read_csv(pvgis_fp, parse_dates=["date"]) 
	pvgis["date"] = pd.to_datetime(pvgis["date"]).dt.normalize()
	# 檢查必要欄位
	if "P_mapped_Wh" not in pvgis.columns:
		raise KeyError("PVGIS 檔案缺少欄位 'P_mapped_Wh'")
	return pvgis


def read_solar(solar_fp: Path) -> pd.DataFrame:
	solar_fp = Path(solar_fp)
	"""
	讀取觀測（SolarRecord）CSV 並回傳標準化欄位。

	輸入檔案通常含 `LocalTime` 和 `Wh`，此函式會把 `LocalTime` 重新命名為 `date`，
	把 `Wh` 重新命名為 `y_obs`，並把 `date` 正規化為只保留日期。

	回傳 DataFrame 含 `date` (datetime) 與 `y_obs`。
	"""
	if not solar_fp.exists():
		raise FileNotFoundError(f"找不到觀測檔案：{solar_fp}")
	solar = pd.read_csv(solar_fp, parse_dates=["LocalTime"]) 
	solar = solar.rename(columns={"LocalTime": "date", "Wh": "y_obs"})
	solar["date"] = pd.to_datetime(solar["date"]).dt.normalize()
	# 檢查必要欄位
	if "y_obs" not in solar.columns:
		raise KeyError("觀測檔案缺少欄位 'Wh'（已重新命名為 y_obs）")
	return solar


def map_pvgis_to_dates(pvgis_df: pd.DataFrame, target_dates) -> pd.Series:
	"""
	把 PVGIS 的 `P_mapped_Wh` 映射到目標日期序列 `target_dates` 上。

	映射邏輯順序：
	1) 若 PVGIS 資料含有目標日期範圍的實際日期，直接以 `merge` 對齊。
	2) 否則先以日序（day-of-year）平均值映射（同年不同年份可沿用日平均）。
	3) 若 DOY 映射仍有缺值，則改用同月份平均值填補。

	輸入：
	- pvgis_df: PVGIS DataFrame，需含 `date` 與 `P_mapped_Wh`。
	- target_dates: 可被 `pd.to_datetime` 解析的日期序列（Index 或 list-like）。

	回傳：對齊到 `target_dates` 的 pd.Series (index = target_dates)，值為 P_mapped_Wh。
	"""
	target_dates = pd.to_datetime(target_dates)
	min_t, max_t = target_dates.min(), target_dates.max()
	# 若 PVGIS 有覆蓋目標日期範圍，則直接合併取得對應的每日值
	mask = (pvgis_df["date"] >= min_t) & (pvgis_df["date"] <= max_t)
	if mask.any():
		sub = pvgis_df.loc[mask, ["date", "P_mapped_Wh"]]
		mapped = pd.DataFrame({"date": target_dates}).merge(sub, on="date", how="left")["P_mapped_Wh"]
		mapped.index = target_dates
		return mapped

	# 否則使用 DOY 映射，再用 month 平均當作退化填補
	tmp = pvgis_df.copy()
	tmp["doy"] = tmp["date"].dt.dayofyear
	doy_map = tmp.groupby("doy")["P_mapped_Wh"].mean()
	doy = pd.DatetimeIndex(target_dates).dayofyear
	mapped_vals = pd.Series(doy).map(doy_map).values
	mapped = pd.Series(mapped_vals, index=target_dates)
	# 若仍有 NaN，改以月份平均值填補
	if mapped.isna().any():
		month_map = tmp.groupby(tmp["date"].dt.month)["P_mapped_Wh"].mean()
		months = pd.DatetimeIndex(target_dates).month
		mapped = mapped.fillna(pd.Series(months, index=target_dates).map(month_map))
	return mapped


def compute_monthly_scaling_k(train_df: pd.DataFrame) -> pd.Series:
	"""
	計算每月縮放係數 k。

	說明：k = sum(obs) / sum(pvgis)（以月份為單位累加），用以把 PVGIS 的每日值尺度調整到觀測值的尺度。

	回傳：一個 index 為 1..12 的 Series，缺值以 1.0 填補（表示不縮放）。
	"""
	if "month" not in train_df.columns:
		train_df = train_df.copy()
		train_df["month"] = train_df["date"].dt.month
	# 若 pvgis 某月總和為 0，避免除以 0，因此先 replace
	pvgis_sum = train_df.groupby("month")["y_pvgis_daily"].sum().replace(0, 1e-6)
	obs_sum = train_df.groupby("month")["y_obs"].sum()
	k = (obs_sum / pvgis_sum).reindex(range(1, 13)).fillna(1.0)
	return k


def prepare_autots_model(forecast_length: int, autots_params: dict | None = None):
	"""
	建立並回傳 AutoTS 模型物件。

	參數:
	- forecast_length: 要預測的天數長度
	- autots_params: 可選的 AutoTS 自訂參數，會覆蓋內部預設值

	此函式使用保守預設值，適合單變數、短序列的殘差預測，並且避免過度搜尋與長時間訓練。
	"""
	try:
		from autots import AutoTS
	except Exception as e:
		raise ImportError("請先安裝 AutoTS：pip install autots") from e

	default_autots = dict(
		forecast_length=forecast_length,
		frequency="D",
		prediction_interval=0.9,
		ensemble=None,
		model_list="superfast",
		transformer_list="superfast",
		max_generations=1,
		num_validations=1,
		n_jobs=1,
		verbose=0,
	)
	if autots_params:
		default_autots.update(autots_params)
	model = AutoTS(**default_autots)
	return model


def fit_autots_residuals(train_long: pd.DataFrame, forecast_length: int, autots_params: dict | None = None):
	"""
	使用 AutoTS 在殘差（單欄 'value'）上進行訓練並產生預測。

	輸入 `train_long` 應為兩欄 DataFrame: `date` 與 `value`（殘差），
	AutoTS 以單序列方式訓練並回傳 `prediction.forecast`（DataFrame）。

	回傳值：
	- model: 訓練後的 AutoTS 模型物件
	- residual_fcst_df: 模型預測的 forecast DataFrame（通常為 1xH 矩陣）
	"""
	model = prepare_autots_model(forecast_length, autots_params)
	# 請注意：AutoTS 的 fit/predict 會回傳複雜的物件，這裡我們只需要 forecast
	model = model.fit(train_long, date_col="date", value_col="value", id_col=None)
	prediction = model.predict()
	residual_fcst_df = prediction.forecast
	return model, residual_fcst_df


def residual_fcst_to_series(residual_fcst_df: pd.DataFrame, start_date: pd.Timestamp, forecast_length: int):
	"""
	把 AutoTS 回傳的 forecast DataFrame 轉為以日期索引的 residual Series。

	說明：AutoTS 的 forecast 可能是矩陣形式（1xH 或 NxH），此函式會把它展平（ravel），
	若實際長度小於指定的 forecast_length，會補 NaN，最後以 0 填補缺值（保護性填法）。

	回傳：
	- residual_series: pd.Series，index = forecast 日期序列
	- fc_index: 對應的 DatetimeIndex
	"""
	vals = np.asarray(residual_fcst_df).ravel()
	# 若模型回傳比要求 shorter，則 pad
	vals = vals[:forecast_length] if len(vals) >= forecast_length else np.concatenate([vals, np.full(forecast_length - len(vals), np.nan)])
	fc_index = pd.date_range(start=start_date, periods=forecast_length, freq="D")
	residual_series = pd.Series(vals, index=fc_index, name="residual_fcst").fillna(0)
	return residual_series, fc_index


def get_pvgis_future_series(pvgis: pd.DataFrame, fc_index: pd.DatetimeIndex) -> pd.Series:
	"""
	取得未來日期範圍內的 PVGIS P_mapped_Wh 序列，若 PVGIS 沒有覆蓋該範圍則使用 map_pvgis_to_dates 做 DOY/月平均填補。

	輸出為與 `fc_index` 對齊的 pd.Series。
	"""
	mask_future = (pvgis["date"] >= fc_index.min()) & (pvgis["date"] <= fc_index.max())
	if mask_future.any():
		pvgis_future = pvgis.loc[mask_future, ["date", "P_mapped_Wh"]].set_index("date").reindex(fc_index)["P_mapped_Wh"]
		if pvgis_future.isna().any():
			pvgis_future = pvgis_future.fillna(map_pvgis_to_dates(pvgis, fc_index))
	else:
		pvgis_future = map_pvgis_to_dates(pvgis, fc_index)
	# 保證回傳的 Series 的 index 與 fc_index 對齊
	return pd.Series(pvgis_future.values, index=fc_index)


def scale_pvgis_future(pvgis_future_series: pd.Series, k: pd.Series) -> pd.Series:
	months_fc = pd.DatetimeIndex(pvgis_future_series.index).month
	scale_vals = pd.Series(months_fc, index=pvgis_future_series.index).map(k).values
	pvgis_future_scaled = pd.Series(pvgis_future_series.values * scale_vals, index=pvgis_future_series.index)
	return pvgis_future_scaled


def compose_forecast(pvgis_future_scaled: pd.Series, residual_series: pd.Series) -> pd.DataFrame:
	y_hat = pvgis_future_scaled + residual_series
	y_hat = y_hat.ffill().fillna(0)
	out_df = pd.DataFrame({
		"date": y_hat.index,
		"y_hat": y_hat.values,
		"pvgis_scaled": pvgis_future_scaled.values,
		"residual_fcst": residual_series.values,
	}).set_index("date")
	return out_df


def save_forecast(out_df: pd.DataFrame, output_csv: Path | str):
	output_csv = Path(output_csv)
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	out_df.to_csv(output_csv)


def plot_actual_vs_pvgis_vs_corrected(
	plot_path,
	index,
	y_actual,
	y_pvgis,
	y_corrected,
	y_naive=None,
	title=None,
	figsize=(12, 6),
	dpi=150,
):
	"""
	繪製 Actual、PVGIS 與校正後預測值的比較圖。

	plot_path: 圖片儲存路徑
	index: DatetimeIndex
	y_actual: 實際觀測值
	y_pvgis: PVGIS P_mapped_Wh 值
	y_corrected: 校正後預測值
	y_naive: 可選的 Naive Lag-1 基準值
	"""
	import matplotlib.pyplot as plt
	import matplotlib.dates as mdates

	plt.figure(figsize=figsize, dpi=dpi)
	plt.plot(index, y_actual, label='Actual', color='black', linewidth=2)
	plt.plot(index, y_pvgis, label='PVGIS P_mapped_Wh', color='dimgray', linewidth=2, linestyle=':')
	plt.plot(index, y_corrected, label='Corrected Forecast', color='tab:blue', linewidth=2)
	if y_naive is not None:
		plt.plot(index, y_naive, label='Naive Lag-1', color='gray', linewidth=1.5, linestyle='--')

	plt.title(title or 'Actual vs PVGIS vs Corrected Forecast')
	plt.xlabel('Date')
	plt.ylabel('Wh')
	plt.grid(alpha=0.3)
	ax = plt.gca()
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
	ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
	plt.xticks(rotation=30)
	plt.legend()
	plt.tight_layout()
	plt.savefig(plot_path, dpi=dpi)
	plt.close()


def compute_forecast_scores(y_true, y_pred, train_series):
	"""
	計算常用的預測評估指標，包含基線比較與規範化誤差。

	參數:
	- y_true: 觀測真實值
	- y_pred: 預測值
	- train_series: 訓練資料序列，用於 MASE 與 RMSSE 的分母計算

	回傳:
	- scores: 字典形式的多項指標
	- mae, mase, rmsse, smape: 其他方便取用的數值
	"""
	import numpy as _np
	from sklearn.metrics import mean_absolute_error as _mae, r2_score as _r2

	y_true = _np.asarray(y_true, dtype=float)
	y_pred = _np.asarray(y_pred, dtype=float)
	train_vals = _np.asarray(train_series, dtype=float)

	mae = float(_mae(y_true, y_pred))
	# MASE: 以訓練資料的一階差分平均絕對值作為基準
	denom = _np.mean(_np.abs(_np.diff(train_vals))) if train_vals.size > 1 else 0.0
	mase = float(mae / denom) if denom != 0 else _np.nan

	rmse = float(_np.sqrt(_np.mean((y_pred - y_true) ** 2)))
	# RMSSE: 以訓練資料的一階差分平方根平均作為分母，提供尺度不變化比較
	denom_rmsse = float(_np.sqrt(_np.mean(_np.diff(train_vals) ** 2))) if train_vals.size > 1 else 0.0
	rmsse = float(rmse / denom_rmsse) if denom_rmsse != 0 else _np.nan

	mean_actual = float(_np.mean(y_true)) if y_true.size > 0 else 0.0
	# nMAE / nRMSE: 以實際平均值做規範化，方便不同量級間比較
	nmae = float(mae / mean_actual) if mean_actual != 0 else _np.nan
	nrmse = float(rmse / mean_actual) if mean_actual != 0 else _np.nan

	smape = float(_np.mean(2.0 * _np.abs(y_pred - y_true) / (_np.abs(y_true) + _np.abs(y_pred) + 1e-9)) * 100)
	# MAPE: 僅對非零實際值計算百分比誤差，避免除以 0
	nonzero_mask = _np.abs(y_true) > 1e-9
	if nonzero_mask.any():
		mape = float(_np.mean(_np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100)
	else:
		mape = _np.nan

	r2 = float(_r2(y_true, y_pred))

	scores = {
		'MAE': mae,
		'MASE_lag1': float(mase) if not _np.isnan(mase) else None,
		'RMSSE': float(rmsse) if not _np.isnan(rmsse) else None,
		'nMAE': float(nmae) if not _np.isnan(nmae) else None,
		'nRMSE': float(nrmse) if not _np.isnan(nrmse) else None,
		'MAPE(%)': float(mape) if not _np.isnan(mape) else None,
		'SMAPE(%)': smape,
		'R2': r2,
	}

	return scores, mae, mase, rmsse, smape


def compute_naive_baseline(train_df, test_df):
	train_wh = train_df['Wh'].astype(float).values
	y_true = test_df['Wh'].astype(float).values
	y_naive = np.r_[train_wh[-1], y_true[:-1]]
	return y_true, y_naive


def load_aligned_pvgis_series(index, pvgis_csv_path):
	"""
	從 PVGIS CSV 讀取基準序列，並依目標日期對齊。
	如果原始資料缺少部分日期，會使用相同月日的平均值補缺。
	"""
	pvgis_csv_path = str(pvgis_csv_path)
	if not Path(pvgis_csv_path).exists():
		raise FileNotFoundError(f'PVGIS file not found: {pvgis_csv_path}')
	pvgis_df = pd.read_csv(pvgis_csv_path, low_memory=True)
	if 'LocalTime' in pvgis_df.columns:
		pvgis_df['LocalTime'] = pd.to_datetime(pvgis_df['LocalTime'], errors='coerce')
		pvgis_df = pvgis_df.set_index('LocalTime')
	elif 'date' in pvgis_df.columns:
		pvgis_df['date'] = pd.to_datetime(pvgis_df['date'], errors='coerce')
		pvgis_df = pvgis_df.set_index('date')
	else:
		raise ValueError('PVGIS CSV missing date index column (LocalTime or date)')

	pvgis_df = pvgis_df.sort_index()
	pvgis_col = 'P_mapped_Wh'
	if pvgis_col not in pvgis_df.columns:
		raise ValueError(f'PVGIS CSV missing column: {pvgis_col}')

	pvgis_series = pvgis_df[pvgis_col]
	aligned = pvgis_series.reindex(index)
	if aligned.isna().all():
		md_map = pvgis_series.groupby(pvgis_series.index.strftime('%m-%d')).mean()
		aligned = pd.Series(index.strftime('%m-%d')).map(md_map).astype(float)
		aligned.index = index
	elif aligned.isna().any():
		md_map = pvgis_series.groupby(pvgis_series.index.strftime('%m-%d')).mean()
		missing = aligned.isna()
		aligned.loc[missing] = pd.Series(index[missing].strftime('%m-%d')).map(md_map).astype(float).values
	return aligned


def save_pvgis_vs_naive_metrics(train_df, test_df, out_dir, horizon, pvgis_csv_path):
	y_true, y_naive = compute_naive_baseline(train_df, test_df)
	pvgis_series = load_aligned_pvgis_series(pd.DatetimeIndex(pd.to_datetime(test_df['date'])), pvgis_csv_path)
	pvgis_values = pvgis_series.astype(float).values

	pvgis_scores, _, _, _, _ = compute_forecast_scores(
		y_true.astype(float),
		pvgis_values,
		train_df['Wh'].astype(float).values,
	)
	naive_scores, _, _, _, _ = compute_forecast_scores(
		y_true.astype(float),
		y_naive.astype(float),
		train_df['Wh'].astype(float).values,
	)

	pvgis_metrics = {
		'PVGIS': pvgis_scores,
		'NaiveLag1': naive_scores,
	}

	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	pvgis_metrics_path = out_dir / f'PVGIS_vs_naiveLag1_metrics_{horizon}d.json'
	with open(pvgis_metrics_path, 'w', encoding='utf-8') as f:
		json.dump(pvgis_metrics, f, ensure_ascii=False, indent=2)
	print('Saved PVGIS vs NaiveLag1 metrics to', pvgis_metrics_path)
	return pvgis_metrics


def residual_correct_with_autots(
	pvgis_fp,
	solar_fp,
	forecast_length=365,
	train_end_date=None,
	autots_params=None,
	output_csv: str | Path | None = None,
	compute_metrics: bool = True,
	metric_horizons: list | None = None,
):
	"""
	殘差校正主流程：
	1. 讀取 PVGIS 與觀測資料
	2. 對齊 PVGIS 到觀測日期
	3. 計算每月縮放係數 k
	4. 建立 residual 序列並訓練 AutoTS
	5. 產生未來 PVGIS + 殘差校正預測
	6. 儲存結果並可選擇計算基準評估指標

	回傳:
	- merged_history: 含觀測、PVGIS、縮放後 PVGIS 與殘差的歷史資料
	- forecast: 最終校正後的預測結果 DataFrame
	- model: 訓練後的 AutoTS 模型物件
	- output_base: 輸出資料夾路徑
	- output_csv: 輸出的 CSV 檔案路徑
	- pvgis_metrics: 若 compute_metrics=True，則回傳的評分結果字典
	"""

	pvgis = read_pvgis(pvgis_fp)
	solar = read_solar(solar_fp)

	# 準備輸出資料夾，結果會依時間戳記分開
	ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	out_base = Path(__file__).parent / "output" / ts
	out_base.mkdir(parents=True, exist_ok=True)
	# 若使用者提供檔名，仍放到 out_base 目錄中；否則使用預設檔名
	if output_csv is None:
		output_csv = out_base / "residual_corrected_forecast.csv"
	else:
		output_csv = out_base / Path(output_csv).name

	# 取出觀測日期與觀測值，並把 PVGIS 映射到這些日期
	obs = solar[["date", "y_obs"]].copy()
	obs_dates = obs["date"]
	obs = obs.set_index("date")
	obs["y_pvgis_daily"] = map_pvgis_to_dates(pvgis, obs_dates)
	obs = obs.reset_index()

	# train_end_date 可選：若指定則只用該日期之前的資料訓練
	if train_end_date is None:
		train = obs.dropna(subset=["y_obs", "y_pvgis_daily"]).copy()
	else:
		train_end_date = pd.to_datetime(train_end_date)
		train = obs[(obs["date"] <= train_end_date) & obs["y_obs"].notna() & obs["y_pvgis_daily"].notna()].copy()
		if train.empty:
			raise ValueError("指定的 train_end_date 導致無訓練資料。")

	# 以 train 資料計算每月縮放係數 k，讓 PVGIS 模擬值得到整體尺度校正
	k = compute_monthly_scaling_k(train)
	obs["month"] = obs["date"].dt.month
	obs["y_pvgis_scaled"] = obs["month"].map(k) * obs["y_pvgis_daily"]
	# 殘差定義：實際觀測值減去縮放後的 PVGIS
	obs["residual"] = obs["y_obs"] - obs["y_pvgis_scaled"]

	# 只用有殘差資料的部分訓練 AutoTS
	train_long = obs[["date", "residual"]].dropna().rename(columns={"residual": "value"})
	if train_long.empty:
		raise ValueError("無可用殘差訓練資料（全部為 NaN）。")

	model, residual_fcst_df = fit_autots_residuals(train_long, forecast_length, autots_params)

	# 將 AutoTS 的殘差預測轉成日期索引的序列
	start = obs["date"].max() + pd.Timedelta(days=1)
	residual_series, fc_index = residual_fcst_to_series(residual_fcst_df, start, forecast_length)

	# 取得未來日期的 PVGIS 基準值，若缺值則用 DOY/月平均補值
	pvgis_future_series = get_pvgis_future_series(pvgis, fc_index)
	# 依照每月縮放係數 k 調整未來 PVGIS，保持與歷史資料同一尺度
	pvgis_future_scaled = scale_pvgis_future(pvgis_future_series, k)

	# 最終預測由校正後 PVGIS 與殘差預測相加組成
	out_df = compose_forecast(pvgis_future_scaled, residual_series)
	save_forecast(out_df, output_csv)
	pvgis_metrics_results = {}
	if compute_metrics:
		if metric_horizons is None:
			metric_horizons = [1, 3, 7, 14, 30, 60, 90, 120 , 150, 180, 365]
		try:
			hist = obs[["date", "y_obs"]].copy()
			hist = hist.rename(columns={"y_obs": "Wh"})
			hist['date'] = pd.to_datetime(hist['date'])
			for h in metric_horizons:
				if len(hist) >= h + 1:
					train_df = hist.iloc[:-h].reset_index(drop=True)
					test_df = hist.iloc[-h:].reset_index(drop=True)
					metrics = save_pvgis_vs_naive_metrics(train_df, test_df, out_base, h, pvgis_csv_path=str(pvgis_fp))
					# compute corrected forecast scores by training AutoTS on residuals from the train subset
					corrected_scores = None
					try:
						# build train/test subsets including PVGIS and residuals from obs
						hist_base = obs[["date", "y_obs", "y_pvgis_daily", "residual"]].copy()
						train_df_base = hist_base.iloc[:-h].reset_index(drop=True)
						test_df_base = hist_base.iloc[-h:].reset_index(drop=True)
						# compute monthly k from train subset
						k_train = compute_monthly_scaling_k(train_df_base)
						# prepare residual training series
						train_long_sub = train_df_base[["date", "residual"]].dropna().rename(columns={"residual": "value"})
						if not train_long_sub.empty:
							model_sub, residual_fcst_df_sub = fit_autots_residuals(train_long_sub, h, autots_params)
							start_sub = pd.to_datetime(train_df_base["date"].max()) + pd.Timedelta(days=1)
							residual_pred_series, fc_index_sub = residual_fcst_to_series(residual_fcst_df_sub, start_sub, h)
							test_idx = pd.DatetimeIndex(pd.to_datetime(test_df_base["date"]))
							pvgis_test = load_aligned_pvgis_series(test_idx, str(pvgis_fp))
							months_test = test_idx.month
							scale_vals = pd.Series(months_test, index=test_idx).map(k_train).values
							pvgis_scaled_test = pvgis_test.values * scale_vals
							residual_pred_aligned = residual_pred_series.reindex(test_idx).fillna(0).values
							corrected_preds = pvgis_scaled_test + residual_pred_aligned
							corrected_scores, *_ = compute_forecast_scores(test_df_base["y_obs"].astype(float).values, corrected_preds.astype(float), train_df_base["y_obs"].astype(float).values)
						try:
							y_true_train = train_df_base["y_obs"].astype(float).values
							y_naive = np.r_[y_true_train[-1], test_df_base["y_obs"].astype(float).values[:-1]]
							plot_path = out_base / f'forecast_vs_actual_vs_naive_lag1_vs_PVGIS_{h}d.png'
							plot_actual_vs_pvgis_vs_corrected(
								plot_path,
								test_idx,
								test_df_base["y_obs"].astype(float).values,
								pvgis_test.values,
								corrected_preds,
								y_naive=y_naive,
								title=f'Actual vs PVGIS vs Corrected Forecast ({h}d)',
							)
						except Exception as e:
							print(f'Failed to save PVGIS comparison plot for {h}d:', e)
					except Exception:
						corrected_scores = None
					# update the saved JSON to include Corrected scores
					try:
						metrics_path = out_base / f'PVGIS_vs_naiveLag1_metrics_{h}d.json'
						with open(metrics_path, 'r', encoding='utf-8') as mf:
							data = json.load(mf)
						data['Corrected'] = corrected_scores
						with open(metrics_path, 'w', encoding='utf-8') as mf:
							json.dump(data, mf, ensure_ascii=False, indent=2)
						pvgis_metrics_results[h] = data
					except Exception:
						pvgis_metrics_results[h] = {'PVGIS': metrics.get('PVGIS'), 'NaiveLag1': metrics.get('NaiveLag1'), 'Corrected': corrected_scores}
		except Exception:
			pvgis_metrics_results = {}

	return {"merged_history": obs, "forecast": out_df, "model": model, "output_base": out_base, "output_csv": Path(output_csv), "pvgis_metrics": pvgis_metrics_results}


if __name__ == "__main__":
	base = Path(__file__).parent / "input"
	pvgis_file = base / "Timeseries_24.148_120.703_E5_0kWp_crystSi_25_35deg_1deg_2005_2005[UTC+8][daily][scaled].csv"
	solar_file = base / "SolarRecord(260228)_d_forWh_WithCodis.csv"

	print("開始：殘差校正流程（AutoTS）")
	res = residual_correct_with_autots(
		pvgis_fp=pvgis_file,
		solar_fp=solar_file,
		forecast_length=90,
		train_end_date=None,
		autots_params={"max_generations": 5, "model_list": "fast"},
	)

	print("已完成，預覽：")
	print(res["merged_history"].head())
	print(res["forecast"].head(10))

