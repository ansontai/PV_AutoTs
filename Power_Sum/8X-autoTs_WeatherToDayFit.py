
import os
import json
import pandas as pd
import numpy as np
from datetime import timedelta


def main():
	base = os.path.dirname(__file__)

	# input CSV (調整路徑到你資料所在位置)
	csv_path = os.path.normpath(os.path.join(base, '..', 'csv', 'forecast_weather_1y.csv'))

	# if not found, try a few common fallback locations and a workspace search
	if not os.path.exists(csv_path):
		candidates = [
			os.path.normpath(os.path.join(base, 'forecast_weather_1y.csv')),
			os.path.normpath(os.path.join(base, '..', 'Power_Sum', 'forecast_weather_1y.csv')),
		]
		found = None
		for c in candidates:
			if os.path.exists(c):
				found = c
				break
		if found is None:
			# search upward from project root (one level above base)
			search_root = os.path.abspath(os.path.join(base, '..'))
			for root, dirs, files in os.walk(search_root):
				if 'forecast_weather_1y.csv' in files:
					found = os.path.join(root, 'forecast_weather_1y.csv')
					break
		if found:
			csv_path = found
		else:
			raise SystemExit(f'Cannot find CSV at {csv_path} (and no fallback found)')

	# try to load data (expect a datetime column named 'LocalTime' or 'date')
	df = pd.read_csv(csv_path, parse_dates=True)
	# try common datetime column names
	dt_col = None
	for c in ('LocalTime', 'local_time', 'date', 'Date', 'datetime'):
		if c in df.columns:
			dt_col = c
			break
	if dt_col is None:
		raise SystemExit('No datetime column found (looked for LocalTime/date/Date/datetime)')

	df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')
	df = df.dropna(subset=[dt_col]).set_index(dt_col).sort_index()

	if 'Wh' not in df.columns:
		raise SystemExit('No Wh column found in CSV')

	# ensure daily frequency
	first, last = df.index.min(), df.index.max()
	full_idx = pd.date_range(start=first, end=last, freq='D')
	df = df.reindex(full_idx)
	df['Wh'] = pd.to_numeric(df['Wh'], errors='coerce').ffill().bfill()

	# AutoTS setup
	try:
		from autots import AutoTS
	except Exception:
		raise SystemExit('autots not installed. pip install autots')

	# forecast_horizon = 365
	forecast_horizon = 365

	# template path (relative to this script). 可修改為你實際位置
	tpl_rel = os.path.normpath(os.path.join(base, 'local', 'autoTs_template', 'autoTs_template_90d.json'))
	if not os.path.exists(tpl_rel):
		# try alternative common path under Power_day output
		alt = os.path.normpath(os.path.join(base, '..', 'Power_day', 'output', 'autoTs_template', 'autoTs_template_90d.json'))
		tpl_path = alt if os.path.exists(alt) else tpl_rel
	else:
		tpl_path = tpl_rel

	# save outputs under the same folder as the input CSV, in an `output/` subfolder
	csv_dir = os.path.dirname(csv_path)
	out_dir = os.path.normpath(os.path.join(csv_dir, 'output', f'7-autoTs_forecast_{forecast_horizon}d'))
	os.makedirs(out_dir, exist_ok=True)

	# instantiate AutoTS with reasonable defaults
	ats = AutoTS(forecast_length=forecast_horizon, frequency='D', model_list='default', n_jobs=4)

	# import template if available
	try:
		if os.path.exists(tpl_path):
			imp = getattr(ats, 'import_template', None)
			if callable(imp):
				imp(tpl_path)
			else:
				# fallback: some versions accept csv/json via same method
				ats.import_template(tpl_path)
			print('Imported template from', tpl_path)
		else:
			print('Template not found, continuing without import')
	except Exception as e:
		print('Template import failed, continuing:', e)

	# fit on all available data and predict future 365 days
	train_wide = df[['Wh']].copy()
	print('Fitting AutoTS on', len(train_wide), 'rows...')
	model = ats.fit(train_wide)

	# predict
	print('Generating forecast...')
	pred = model.predict()
	forecast = pred.forecast

	# align index to future days
	last_date = df.index.max()
	future_index = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon, freq='D')
	try:
		forecast.index = future_index
	except Exception:
		pass

	out_csv = os.path.join(out_dir, f'forecast_Wh_autots_{forecast_horizon}d.csv')
	forecast.to_csv(out_csv, index=True)
	print('Forecast saved to', out_csv)

	# save model results if possible
	try:
		res = model.results()
		res_path = os.path.join(out_dir, f'autots_model_results_{forecast_horizon}d.json')
		with open(res_path, 'w', encoding='utf-8') as f:
			json.dump(res, f, ensure_ascii=False, indent=2)
		print('Model results saved to', res_path)
	except Exception as e:
		print('Could not save model.results():', e)

	# export template of best model
	try:
		tpl_out_dir = os.path.join(out_dir, 'autoTs_template')
		os.makedirs(tpl_out_dir, exist_ok=True)
		tpl_out = os.path.join(tpl_out_dir, f'autoTs_template_{forecast_horizon}d.csv')
		export = getattr(model, 'export_template', None)
		if callable(export):
			model.export_template(tpl_out, models='best', n=1, max_per_model_class=1, include_results=True)
			print('Exported template to', tpl_out)
	except Exception as e:
		print('Template export failed:', e)


if __name__ == '__main__':
	main()

