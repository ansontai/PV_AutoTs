import pandas as pd
import sys
import re
from pathlib import Path
import numpy as np


def read_pvgis_tmy(path):
	# 讀整個檔案前段：找出 header (time...)、month->year mapping、以及 lat/lon metadata
	lat = lon = None
	header_line = None
	mapping = {}
	with open(path, 'r', encoding='utf-8') as f:
		lines = f.readlines()

		# 找 header 行（time(UTC) 等）
		for i, line in enumerate(lines):
			if re.match(r'^\s*time(\(|,|\b)', line.strip().lower()):
				header_line = i
				break

		# 解析 header 之前的 month,year 區塊
		if header_line is not None:
			for j, line in enumerate(lines[:header_line]):
				if re.match(r'^\s*month\s*,\s*year\s*$', line.strip().lower()):
					for k in range(j+1, header_line):
						ln = lines[k].strip()
						m = re.match(r'^(\d{1,2})\s*,\s*(\d{4})\s*$', ln)
						if m:
							mo = int(m.group(1))
							yr = int(m.group(2))
							mapping[mo] = yr
						else:
							continue
					break

		# 嘗試擷取 lat/lon（在 header 之前的 metadata 行）
		for line in lines[:header_line if header_line is not None else None]:
			low = line.strip().lower()
			if 'latitude' in low and lat is None:
				m = re.search(r'[-+]?[0-9]*\.?[0-9]+', line)
				if m:
					try:
						lat = float(m.group(0))
					except Exception:
						lat = None
			if 'longitude' in low and lon is None:
				m = re.search(r'[-+]?[0-9]*\.?[0-9]+', line)
				if m:
					try:
						lon = float(m.group(0))
					except Exception:
						lon = None

	if header_line is None:
		raise RuntimeError('找不到資料表頭 (time...)')

	# 以該行為 header 讀入，再手動將第一欄轉成 datetime（避免 date_parser 相容性問題）
	df = pd.read_csv(path, header=header_line)
	# 第一欄名可能包含括號，統一命名為 `time_utc`
	time_col = df.columns[0]
	df.rename(columns={time_col: 'time_utc'}, inplace=True)

	# 有些 metadata 行可能仍被讀成資料列（如 "T2m: 2-m air temperature (degree Celsius)"）
	# 先過濾出符合 YYYYMMDD:HHMM 格式的列
	df['time_utc'] = df['time_utc'].astype(str)
	mask = df['time_utc'].str.match(r'^\s*\d{8}:\d{4}\s*$')
	df = df.loc[mask].copy()

	# 範例時間格式: 20180101:0000
	# 在轉成 datetime 前，若有 month->year mapping，將時間字串中的 year 替換為 mapping 指定的 year
	df['time_utc_str'] = df['time_utc'].str.strip()
	if mapping:
		def _replace_year(ts):
			m = re.match(r'^(\d{4})(\d{2})(\d{2}:\d{4})$', ts)
			if not m:
				return ts
			yr, mo, rest = m.group(1), m.group(2), m.group(3)
			mo_i = int(mo)
			mapped = mapping.get(mo_i)
			if mapped:
				return f"{mapped}{mo}{rest}"
			return ts
		df['time_utc_str'] = df['time_utc_str'].apply(_replace_year)

	# 轉換為 timezone-aware datetime（原始是 UTC+0）
	df['time_utc'] = pd.to_datetime(df['time_utc_str'], format='%Y%m%d:%H%M', utc=True, errors='coerce')
	df = df.dropna(subset=['time_utc']).copy()

	# 若有 mapping，將所有時間投射到單一年份（最小 mapping 年）以避免 resample 跨年空白區塊
	if mapping:
		target_year = min(mapping.values())
		def _year_align(ts):
			return ts.tz_convert(None).replace(year=target_year).tz_localize('UTC')
		df['time_utc'] = df['time_utc'].apply(_year_align)

	# 把欄位轉 UTC+8
	df['time_utc'] = df['time_utc'].dt.tz_convert('Asia/Taipei')

	df = df.set_index('time_utc')
	# 移除輔助欄位
	df = df.drop(columns=['time_utc_str'], errors='ignore')
	return df, lat, lon


def compute_solar_elevation(df_index, lat, lon):
	# 優先使用 pvlib 計算，若無 pvlib 則用 G(h)>0 取代（日照判斷）
	try:
		import pvlib
		sp = pvlib.solarposition.get_solarposition(df_index, latitude=lat, longitude=lon)
		# solar elevation angle in degrees
		return sp['apparent_elevation']
	except Exception:
		return None


def circular_mean_deg(deg, w=None):
	"""計算風向圓形平均（度），可選擇以風速加權。
	deg: pandas Series of degrees (可包含 NaN)
	w: pandas Series 或 array-like 的權重（與 deg index 對齊）
	"""
	# 處理輸入
	if deg is None:
		return np.nan
	deg = deg.dropna()
	if deg.empty:
		return np.nan
	rad = np.deg2rad(deg.to_numpy())
	if w is None:
		w_arr = np.ones_like(rad)
	else:
		# 讓權重與 deg 對齊
		try:
			w_arr = np.asarray(w.loc[deg.index])
		except Exception:
			w_arr = np.asarray(w)
		# 避免全 0 權重
		if np.all(w_arr == 0):
			w_arr = np.ones_like(w_arr)

	x = np.sum(w_arr * np.cos(rad))
	y = np.sum(w_arr * np.sin(rad))
	ang = np.arctan2(y, x)
	return (np.rad2deg(ang) + 360) % 360


def circular_std_deg(deg, w=None):
	"""計算風向圓形標準差（度），權重可選。
	參考公式：std = sqrt(-2 * ln(R))，其中 R 為單位向量長度（加權後）
	返回度數。
	"""
	if deg is None:
		return np.nan
	deg = deg.dropna()
	if deg.empty:
		return np.nan
	rad = np.deg2rad(deg.to_numpy())
	if w is None:
		w_arr = np.ones_like(rad)
	else:
		try:
			w_arr = np.asarray(w.loc[deg.index])
		except Exception:
			w_arr = np.asarray(w)
		if np.all(w_arr == 0):
			w_arr = np.ones_like(w_arr)

	x = np.sum(w_arr * np.cos(rad))
	y = np.sum(w_arr * np.sin(rad))
	R = np.sqrt(x * x + y * y) / np.sum(w_arr)
	R = np.clip(R, 1e-12, 1.0)
	std_rad = np.sqrt(-2.0 * np.log(R))
	return np.rad2deg(std_rad)


def aggregate_daily(df, lat=None, lon=None):
	# 以建議欄位集做日聚合，並使用一致的欄名後綴
	# 移除不需要的欄位
	drop_cols = [c for c in ['Int'] if c in df.columns]
	df = df.drop(columns=drop_cols, errors='ignore')

	# 將欄位轉為數值型（非數值會變為 NaN），以便後續聚合
	df = df.apply(pd.to_numeric, errors='coerce')

	# 計算太陽高度（如果提供 lat/lon）
	solar_elev = None
	if lat is not None and lon is not None:
		solar_elev = compute_solar_elevation(df.index, lat, lon)
		if solar_elev is not None:
			df = df.copy()
			df['H_sun'] = solar_elev.values

	# 估算每筆間隔（小時），用於從功率/輻照轉能量
	try:
		dt_hours = df.index.to_series().diff().dt.total_seconds().median() / 3600.0
		if pd.isna(dt_hours) or dt_hours <= 0:
			dt_hours = 1.0
	except Exception:
		dt_hours = 1.0

	# 每日樣本數（用於 valid fraction）
	counts_per_day = df.resample('D').size()

	# 建立結果 dataframe（以 dates 為 index）
	daily = pd.DataFrame(index=counts_per_day.index)

	# 能量 / 輻照類欄位（以 Wh / Wh/m2 為單位）
	irr_cols = ["G(i)", "G(h)", "Gb(n)", "Gd(h)", "IR(h)"]
	for c in irr_cols:
		if c in df.columns:
			# 每筆量為 value * dt_hours -> Wh 或 Wh/m2
			daily[c + "_Whm2"] = (df[c] * dt_hours).resample('D').sum()
			daily[c + "_kWhm2"] = daily[c + "_Whm2"] / 1000.0
			daily[c + "_mean"] = df[c].resample('D').mean()
			daily[c + "_min"] = df[c].resample('D').min()
			daily[c + "_max"] = df[c].resample('D').max()
			daily[c + "_std_Wm2"] = df[c].resample('D').std()
			daily[c + "_p10_Wm2"] = df[c].resample('D').quantile(0.1)
			daily[c + "_p90_Wm2"] = df[c].resample('D').quantile(0.9)

	# 功率 -> 能量
	if 'P' in df.columns:
		daily['P_Wh'] = (df['P'] * dt_hours).resample('D').sum()
		daily['P_kWh'] = daily['P_Wh'] / 1000.0
		daily['P_mean_W'] = df['P'].resample('D').mean()
		daily['P_min_W'] = df['P'].resample('D').min()
		daily['P_max_W'] = df['P'].resample('D').max()
		daily['P_std_W'] = df['P'].resample('D').std()
		daily['P_p10_W'] = df['P'].resample('D').quantile(0.1)
		daily['P_p90_W'] = df['P'].resample('D').quantile(0.9)

	# 氣象欄位: 均值/極值/變異
	if "T2m" in df.columns:
		daily["T2m_mean"] = df["T2m"].resample("D").mean()
		daily["T2m_min"]  = df["T2m"].resample("D").min()
		daily["T2m_max"]  = df["T2m"].resample("D").max()
		daily["T2m_std"] = df["T2m"].resample("D").std()
		daily["T2m_p10"] = df["T2m"].resample("D").quantile(0.1)
		daily["T2m_p90"] = df["T2m"].resample("D").quantile(0.9)

	if "RH" in df.columns:
		daily["RH_mean"] = df["RH"].resample("D").mean()
		daily["RH_min"]  = df["RH"].resample("D").min()
		daily["RH_max"]  = df["RH"].resample("D").max()
		daily["RH_std"] = df["RH"].resample("D").std()
		daily["RH_p10"] = df["RH"].resample("D").quantile(0.1)
		daily["RH_p90"] = df["RH"].resample("D").quantile(0.9)

	if "WS10m" in df.columns:
		daily["WS10m_mean"] = df["WS10m"].resample("D").mean()
		daily["WS10m_max"]  = df["WS10m"].resample("D").max()
		daily["WS10m_std"] = df["WS10m"].resample("D").std()
		daily["WS10m_p10"] = df["WS10m"].resample("D").quantile(0.1)
		daily["WS10m_p90"] = df["WS10m"].resample("D").quantile(0.9)

	if "SP" in df.columns:
		daily["SP_mean"] = df["SP"].resample("D").mean()

	# 太陽高度 / 日照
	if 'H_sun' in df.columns:
		daily['H_sun_max'] = df['H_sun'].resample('D').max()
		daily['H_sun_mean'] = df['H_sun'].resample('D').mean()

	# 風向：圓形平均、圓形標準差與主導扇區
	if "WD10m" in df.columns:
		def _wd_circ_mean(s):
			w = df["WS10m"] if "WS10m" in df.columns else None
			return circular_mean_deg(s, w=w)
		def _wd_circ_std(s):
			w = df["WS10m"] if "WS10m" in df.columns else None
			return circular_std_deg(s, w=w)
		def _wd_mode_sector(s):
			degs = s.dropna()
			if degs.empty:
				return np.nan
			sectors = ((degs + 22.5) // 45).astype(int) % 8
			labels = ['N','NE','E','SE','S','SW','W','NW']
			mode_idx = sectors.value_counts().idxmax()
			return labels[int(mode_idx)]
		daily["WD10m_circmean"] = df["WD10m"].resample("D").apply(_wd_circ_mean)
		daily["WD10m_circstd_deg"] = df["WD10m"].resample("D").apply(_wd_circ_std)
		daily["WD10m_mode_sector"] = df["WD10m"].resample("D").apply(_wd_mode_sector)

	# 觀測數與有效比例
	key_cols = [c for c in ['P','G(i)','G(h)','T2m','RH','WS10m','WD10m','SP','H_sun'] if c in df.columns]
	for c in key_cols:
		n_obs = df[c].resample('D').count()
		daily[f'n_obs_{c}'] = n_obs
		with np.errstate(divide='ignore', invalid='ignore'):
			daily[f'valid_frac_{c}'] = n_obs / counts_per_day.replace(0, np.nan)

	# 標準化 index 為當日 00:00 並新增 date 欄位
	daily.index = daily.index.normalize()
	try:
		daily['date'] = daily.index.strftime('%Y-%m-%d')
	except Exception:
		daily['date'] = pd.to_datetime(daily.index).strftime('%Y-%m-%d')

	# 把 date 移到第一欄
	cols = ['date'] + [c for c in daily.columns if c != 'date']
	daily = daily[cols]

	return daily


def main(in_path, out_path=None):
	df, lat, lon = read_pvgis_tmy(in_path)
	daily = aggregate_daily(df, lat=lat, lon=lon)
	if out_path is None:
		out_path = Path(in_path).with_name(Path(in_path).stem + '-daily.csv')
	out_path = Path(out_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	# 在輸出前新增 `DATE` 欄並移到第一欄（格式 YYYY-MM-DD），不輸出 index
	try:
		date_series = daily.index.strftime('%Y-%m-%d')
	except Exception:
		date_series = pd.to_datetime(daily.index).strftime('%Y-%m-%d')
	# 移除可能存在的 大寫 'DATE' 欄位
	if 'DATE' in daily.columns:
		daily = daily.drop(columns=['DATE'])
	# 將小寫 'date' 欄設定為 YYYY-MM-DD（覆寫或新增），並移到第一欄
	if 'date' in daily.columns:
		daily['date'] = date_series
		cols = ['date'] + [c for c in daily.columns if c != 'date']
		daily = daily[cols]
	else:
		daily.insert(0, 'date', date_series)

	daily.to_csv(out_path, index=False)
	print('輸出：', out_path)
	print('輸出：', out_path)


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('用法: python PVGIS_TmyCsv_handle.py <input.csv> [output.csv]')
		sys.exit(1)
	inp = sys.argv[1]
	outp = sys.argv[2] if len(sys.argv) > 2 else None
	main(inp, outp)

