"""只針對 SunShine / SunshineRate 的快速 AutoTS 預測。

輸入:
	csv/2000--202602-d-forWh.csv

輸出:
	output/forecast_SunShine_SunshineRate_autots.csv
"""

from pathlib import Path

import pandas as pd

try:
	from autots import AutoTS
except ImportError as exc:
	raise ImportError("需要安裝 autots：pip install autots") from exc



def forecast_one_column(
	df: pd.DataFrame,
	column: str,
	forecast_length: int,
	model_kwargs: dict,
) -> pd.DataFrame:
	"""Train one AutoTS model and return Date + forecast value for one column."""
	print(f"\n==== {column} 原始資料分布 ====")
	print(df[column].describe())
	print(df[column].value_counts(dropna=False).head(10))
	ts = df[["Date", column]].copy()
	ts[column] = pd.to_numeric(ts[column], errors="coerce")
	ts = ts.dropna()
	if ts.empty:
		raise ValueError(f"欄位 {column} 沒有可用數值資料")

	ts = ts.rename(columns={"Date": "ds", column: "y"})

	model = AutoTS(**model_kwargs)
	model = model.fit(ts, date_col="ds", value_col="y")
	pred = model.predict(forecast_length=forecast_length).forecast

	out = pred.reset_index().rename(columns={"index": "Date"})
	if "y" in out.columns:
		out = out.rename(columns={"y": column})
	else:
		value_cols = [c for c in out.columns if c != "Date"]
		if not value_cols:
			raise ValueError(f"欄位 {column} 的預測結果缺少數值欄")
		out = out.rename(columns={value_cols[0]: column})

	out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
	out = out.dropna(subset=["Date"])
	return out[["Date", column]]


def main() -> None:
	workspace_dir = Path(__file__).resolve().parents[1]
	input_path = workspace_dir / "csv" / "2000--202602-d-forWh.csv"
	output_dir = workspace_dir / "autoTs_weather" / "output"
	output_dir.mkdir(exist_ok=True)
	output_path = output_dir / "forecast_SunShine_SunshineRate_autots.csv"

	if not input_path.exists():
		raise FileNotFoundError(f"找不到輸入檔案: {input_path}")

	print(f"讀取資料: {input_path}")
	df = pd.read_csv(input_path)

	if "Date" not in df.columns:
		if all(c in df.columns for c in ("Year", "Month", "Day")):
			df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]], errors="coerce")
		else:
			raise ValueError("缺少 Date 欄位，且無 Year/Month/Day 可組合日期")
	else:
		df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

	df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date")

	target_cols = ["SunShine", "SunshineRate"]
	missing = [c for c in target_cols if c not in df.columns]
	if missing:
		raise ValueError(f"輸入檔缺少欄位: {missing}")

	# 最快速設定：僅使用 superfast 模型池、單次世代與單次驗證。
	forecast_length = 365
	model_kwargs = {
		"forecast_length": forecast_length,
		"frequency": "D",
		"model_list": "superfast",
    #"model_list": ["AverageValueNaive"], ### 測試用最簡單模型，確保流程順暢
		#"ensemble": "simple", ### superfast 模型池已經不包含需要 torch 的模型了，應該不太會有相關錯誤了，但還是保守起見不使用 horizontal ensemble
    "ensemble": "auto",
		"max_generations": 15,
		"num_validations": 2,
		"prediction_interval": 0.9,
	}

	print("開始訓練與預測: SunShine, SunshineRate")
	forecast_sunshine = forecast_one_column(df, "SunShine", forecast_length, model_kwargs)
	forecast_sunshine_rate = forecast_one_column(df, "SunshineRate", forecast_length, model_kwargs)

	merged = pd.merge(forecast_sunshine, forecast_sunshine_rate, on="Date", how="outer")
	merged = merged.sort_values("Date")
	merged.to_csv(output_path, index=False)

	print(f"輸出完成: {output_path}")
	print(merged.head())


if __name__ == "__main__":
	main()
