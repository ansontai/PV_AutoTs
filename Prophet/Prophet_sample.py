"""
最小 Prophet 範例

說明:
- 安裝: `pip install prophet pandas numpy`
- 執行: `python Prophet_sample.py`

此範例會建立合成時間序列、訓練 Prophet、並輸出未來 30 天的預測。
"""

import numpy as np
import pandas as pd
from prophet import Prophet


def make_sample_df():
	np.random.seed(42)
	dates = pd.date_range(start="2020-01-01", periods=365, freq="D")
	# 合成訊號：基線 + 年週期 + 小量雜訊
	seasonal = 5.0 * np.sin(2 * np.pi * dates.dayofyear / 365.0)
	trend = 0.01 * np.arange(len(dates))
	noise = np.random.normal(scale=1.0, size=len(dates))
	y = 10.0 + trend + seasonal + noise
	return pd.DataFrame({"ds": dates, "y": y})


def main():
	df = make_sample_df()

	m = Prophet()
	m.fit(df)

	future = m.make_future_dataframe(periods=30)
	forecast = m.predict(future)

	# 印出未來 30 天的預測（yhat, 下界, 上界）
	print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30).to_string(index=False))


if __name__ == "__main__":
	main()

