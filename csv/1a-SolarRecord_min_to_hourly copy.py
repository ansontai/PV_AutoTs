# 將分鐘級資料重新採樣為小時級，缺值用同小時全日平均補齊
# 時間格式也會整理成 pandas 可識別的 datetime 格式
import pandas as pd
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / 'SolarRecord_260310_1829-fixed_0b.csv'
OUTPUT_CSV = SCRIPT_DIR / 'SolarRecord_260310_1829-hour.csv'

def main():
	# 讀取資料
	# 先偵測所有 header row（以 'LocalTime' 開頭）
	with open(INPUT_CSV, 'r', encoding='utf-8') as f:
		lines = f.readlines()
	# 只保留第一個 header，後續遇到 header row 直接略過
	header = None
	data_lines = []
	for line in lines:
		if line.strip().startswith('LocalTime'):
			if header is None:
				header = line
			continue
		data_lines.append(line)
	# 用第一個 header 作為欄位名稱
	from io import StringIO
	csv_content = header + ''.join(data_lines)
	df = pd.read_csv(StringIO(csv_content))
	# 過濾掉中間混入的 header row
	df = df[df['LocalTime'] != 'LocalTime']
	# 轉換 LocalTime 為 datetime，允許多種格式
	df['LocalTime'] = pd.to_datetime(df['LocalTime'], format='mixed', errors='coerce')
	df = df.set_index('LocalTime')

	# 嘗試將所有欄位轉為數值，無法轉的會成為 NaN，避免字串造成 mean() 失敗
	df = df.apply(lambda s: pd.to_numeric(s, errors='coerce'))

	# 以每小時為單位重新採樣（平均）
	hourly = df.resample('h').mean()

	# 找出缺值（NaN）的位置
	nan_mask = hourly.isna()
	# 取得每筆資料的「小時」欄位（0~23）
	hourly['hour'] = hourly.index.hour

	# 用所有天同一小時的平均值補齊缺值
	for col in hourly.columns:
		if col == 'hour':
			continue
		# 依小時分組計算平均
		hour_means = hourly.groupby('hour')[col].transform('mean')
		# 補齊缺值
		hourly[col] = hourly[col].fillna(hour_means)

	# 移除輔助欄位
	# hourly = hourly.drop(columns=['hour'])

	# 輸出結果
	OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
	hourly.to_csv(OUTPUT_CSV, float_format='%.5f')

if __name__ == '__main__':
	main()
