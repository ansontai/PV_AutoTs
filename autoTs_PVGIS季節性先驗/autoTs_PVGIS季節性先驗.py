# AutoTS training + forecast using SolarRecord daily Wh and TMY seasonal prior.
# Save this file as autoTs_PVGIS季節性先驗.py under the folder autoTs_PVGIS季節性先驗/

import os
from pathlib import Path
import pandas as pd
import numpy as np
from autots import AutoTS
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_target(solar_path):
	df = pd.read_csv(solar_path, parse_dates=["LocalTime"], dayfirst=False)
	df = df.set_index("LocalTime").sort_index()
	df = df[["Wh"]].rename(columns={"Wh": "y"})
	df["y"] = pd.to_numeric(df["y"], errors="coerce")
	df = df.dropna(subset=["y"])
	return df


def build_tmy_prior_from_daily_csv(tmy_path, target_index, prefer_cols=("G(h)_Whm2","T2m_mean","RH_mean")):
	tmy = pd.read_csv(tmy_path, parse_dates=["date"]).set_index("date").sort_index()
	available = [c for c in prefer_cols if c in tmy.columns]
	if not available:
		raise ValueError(f"TMY CSV 找不到欄位，候選:{prefer_cols}，實際:{list(tmy.columns)}")
	tmy_small = tmy[available].copy()
	# rename columns to tmy_*
	col_map = {}
	for c in available:
		if "G(h)" in c:
			col_map[c] = "tmy_G_whm2"
		elif "T2m" in c:
			col_map[c] = "tmy_T2m_mean"
		elif "RH" in c:
			col_map[c] = "tmy_RH_mean"
		else:
			col_map[c] = "tmy_" + c
	tmy_small = tmy_small.rename(columns=col_map)

	# use month-day (MM-DD) to map daily TMY onto target index
	tmy_small["md"] = tmy_small.index.strftime("%m-%d")
	tmy_by_md = tmy_small.groupby("md").mean()
	keys = pd.DatetimeIndex(target_index).strftime("%m-%d")
	prior = tmy_by_md.reindex(keys).reset_index(drop=True)
	prior.index = pd.DatetimeIndex(target_index)

	# fill Feb-29 using Feb-28 if present
	if prior.isna().any().any():
		idx = pd.DatetimeIndex(target_index)
		keys_fix = np.where((idx.month == 2) & (idx.day == 29), "02-28", idx.strftime("%m-%d"))
		prior = tmy_by_md.reindex(keys_fix)
		prior.index = idx

	prior.columns = [c if c.startswith("tmy_") else "tmy_" + c for c in prior.columns]
	return prior


def get_paths(base=None):
	if base is None:
		base = Path(__file__).resolve().parent
	input_dir = base / "input"
	solar_path = input_dir / "SolarRecord(260228)_d_forWh_WithCodis.csv"
	tmy_path = input_dir / "tmy_24.148_120.703_2005_2023[UTC+8][daily].csv"
	# prefer folder named 'ouput' (common typo) if present, else use 'output'
	if (base / "ouput").exists():
		out_dir = base / "ouput"
	else:
		out_dir = base / "output"
	out_dir.mkdir(parents=True, exist_ok=True)
	return base, input_dir, solar_path, tmy_path, out_dir


def prepare_data(solar_path, tmy_path):
	df = load_target(solar_path)
	prior = build_tmy_prior_from_daily_csv(tmy_path, df.index)
	df_wide = df.join(prior, how="left")
	# use .ffill() for forward-fill (pandas 3.x removed fillna(method=...))
	df_wide = df_wide.ffill().fillna(df_wide.mean())
	return df_wide


def build_weights(df_wide, y_weight=20):
	weights = {"y": y_weight}
	for c in df_wide.columns:
		if c != "y":
			weights[c] = 1
	return weights


def split_train_test(df_wide, forecast_length=30, min_extra=10):
	if len(df_wide) <= forecast_length + min_extra:
		raise SystemExit("資料太短，請增加歷史資料或減少 forecast_length。")
	train = df_wide.iloc[:-forecast_length]
	test = df_wide.iloc[-forecast_length:]
	return train, test


def build_autots_model(forecast_length=30):
	return AutoTS(
		forecast_length=forecast_length,
		frequency="D",
		ensemble="simple",
		max_generations=5,
		num_validations=2,
		validation_method="backwards",
		model_list="fast",
	)


def fit_and_predict(model, train, weights):
	print("Start fitting AutoTS (this may take some time)...")
	model = model.fit(train, weights=weights)
	pred = model.predict()
	return model, pred


def extract_yhat(pred, train_index, test_index):
	forecast_df = None
	attr = getattr(pred, "forecast", None)
	if attr is not None:
		forecast_df = attr
	elif isinstance(pred, dict) and "forecast" in pred:
		forecast_df = pred.get("forecast")
	elif isinstance(pred, pd.DataFrame):
		# some AutoTS versions return a DataFrame directly
		forecast_df = pred

	if forecast_df is None:
		raise RuntimeError("無法從 AutoTS.predict() 取得 forecast。請檢查 AutoTS 版本。")

	yhat = forecast_df[["y"]].copy()
	if len(yhat) == len(test_index):
		yhat.index = test_index
	else:
		start = train_index[-1] + pd.Timedelta(days=1)
		yhat.index = pd.date_range(start=start, periods=len(yhat), freq="D")
	return yhat


def evaluate_forecast(yhat, test):
	mae = mean_absolute_error(test["y"], yhat["y"])
	mse = mean_squared_error(test["y"], yhat["y"])
	rmse = np.sqrt(mse)
	print(f"MAE={mae:.4f}, RMSE={rmse:.4f}")
	return mae, rmse


def save_forecast(yhat, out_dir, name="yhat_autots.csv"):
	out_path = os.path.join(out_dir, name)
	yhat.to_csv(out_path)
	print("預測已儲存：", out_path)
	return out_path


def main():
	base, input_dir, solar_path, tmy_path, out_dir = get_paths()

	df_wide = prepare_data(solar_path, tmy_path)
	weights = build_weights(df_wide, y_weight=20)

	forecast_length = 30
	train, test = split_train_test(df_wide, forecast_length=forecast_length)

	model = build_autots_model(forecast_length=forecast_length)
	model, pred = fit_and_predict(model, train, weights)

	yhat = extract_yhat(pred, train.index, test.index)

	evaluate_forecast(yhat, test)
	save_forecast(yhat, out_dir)


if __name__ == "__main__":
	main()

