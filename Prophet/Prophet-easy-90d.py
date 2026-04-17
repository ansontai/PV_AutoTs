#!/usr/bin/env python3
"""最簡單的 Prophet 範例：
讀取專案中的日別 Wh 時序資料，使用 Prophet 做 90 天預測，並輸出 CSV。
"""
from __future__ import annotations
import os
import sys
import pandas as pd


def find_input_file(base: str):
	candidates = [
		os.path.join(base, '..', 'csv', '2000--202602-d-forWh_4b.csv'),
		os.path.join(base, '..', 'csv', '2000--202602-d-forWh.csv'),
		os.path.join(base, '..', 'csv', 'SolarRecord_260310_1829-daily-1d.csv'),
		os.path.join(base, 'csv', '2000--202602-d-forWh_4b.csv'),
		os.path.join(base, 'csv', '2000--202602-d-forWh.csv'),
		os.path.join(base, 'SolarRecord(260204).csv'),
		os.path.join(base, 'csv', 'SolarRecord(260204).csv'),
	]
	for p in candidates:
		if os.path.exists(p):
			return p
	return None


def load_series(path: str):
	df = pd.read_csv(path, low_memory=False)
	# try common datetime column names
	date_cols = [c for c in df.columns if c.lower() in ('localtime', 'date', 'time', 'timestamp')]
	if date_cols:
		dtc = date_cols[0]
		try:
			df[dtc] = pd.to_datetime(df[dtc], errors='coerce')
			df = df.set_index(dtc)
		except Exception:
			pass
	else:
		first = df.columns[0]
		try:
			parsed = pd.to_datetime(df[first], errors='coerce')
			if parsed.notna().sum() > 0:
				df[first] = parsed
				df = df.set_index(first)
		except Exception:
			pass

	# pick the target column 'Wh' if present
	if 'Wh' in df.columns:
		ser = df[['Wh']].copy()
	else:
		numcols = df.select_dtypes(include='number').columns.tolist()
		if not numcols:
			raise SystemExit('找不到可用的數值欄位來預測 (例如 Wh)')
		ser = df[[numcols[0]]].copy()
	ser = ser.apply(pd.to_numeric, errors='coerce')
	ser = ser.dropna(how='all')
	try:
		ser.index = pd.to_datetime(ser.index)
		ser = ser.sort_index()
	except Exception:
		pass
	return ser


def main():
	base = os.path.dirname(__file__)
	inp = find_input_file(base)
	if inp is None:
		print('未找到預設 CSV。請將資料放在 csv/ 底下，或修改腳本路徑。')
		sys.exit(2)
	print('Using input:', inp)

	ser = load_series(inp)
	if ser.empty:
		print('讀入時序資料為空，停止。')
		sys.exit(2)

	try:
		from prophet import Prophet
	except Exception:
		try:
			from fbprophet import Prophet
		except Exception:
			print('Prophet 尚未安裝，請執行: pip install prophet  或 pip install fbprophet')
			raise

	# Prepare dataframe for Prophet: columns 'ds' and 'y'
	dfp = ser.reset_index()
	dfp.columns = ['ds', 'y']
	dfp['ds'] = pd.to_datetime(dfp['ds'], errors='coerce')
	dfp = dfp.dropna(subset=['ds'])

	m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
	m.fit(dfp)

	future = m.make_future_dataframe(periods=90, freq='D')
	forecast = m.predict(future)

	# save only the forecast tail (the future periods)
	out_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index('ds')
	out_path = os.path.join(base, 'prophet_minimal_forecast_90d.csv')
	out_forecast.to_csv(out_path)
	print('Saved forecast to', out_path)


if __name__ == '__main__':
	main()

