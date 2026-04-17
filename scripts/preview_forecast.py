from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

workspace = Path(__file__).resolve().parents[1]
out_dir = workspace / 'output'
forecast_csv = out_dir / 'forecast_Wh_autots_30d.csv'
actual_csv = workspace / 'csv' / 'SolarRecord(260204)_d_forWh_WithCodis.csv'
preview_path = out_dir / 'preview_forecast_30d.png'

if not forecast_csv.exists():
    raise SystemExit(f'forecast CSV not found: {forecast_csv}')
if not actual_csv.exists():
    raise SystemExit(f'actual CSV not found: {actual_csv}')

# read forecast
fc = pd.read_csv(forecast_csv, index_col=0, parse_dates=True)
# pick first numeric column as forecast
if 'Wh' in fc.columns:
    y_pred = fc['Wh'].astype(float).values
else:
    y_pred = fc.iloc[:, 0].astype(float).values
# forecast index
try:
    dates = pd.to_datetime(fc.index)
except Exception:
    dates = pd.to_datetime(fc.iloc[:,0])

# read actuals
act = pd.read_csv(actual_csv, parse_dates=['LocalTime'])
act = act.dropna(subset=['LocalTime']).set_index('LocalTime').sort_index()
# resample daily and take Wh
if 'Wh' not in act.columns:
    raise SystemExit('No Wh column in actual CSV')
act_daily = act['Wh'].resample('D').sum().ffill()

# align actual to forecast dates
actual_vals = act_daily.reindex(dates).ffill().fillna(0).astype(float).values

# create preview plot (small)
plt.figure(figsize=(6,3), dpi=150)
plt.plot(dates, actual_vals, label='Actual', color='black', linewidth=2)
plt.plot(dates, y_pred, label='Forecast', color='dimgray', linewidth=2)
plt.plot(dates, np.concatenate(([act_daily.iloc[-1]], actual_vals[:-1])), label='Naive Lag-1', color='gray', linewidth=1.5, linestyle='--')
plt.title('Preview: Wh Forecast vs Actual (30d)')
plt.xlabel('Date')
plt.ylabel('Wh')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
plt.xticks(rotation=30)
plt.tight_layout()
plt.legend()
plt.savefig(preview_path, bbox_inches='tight')
print('Preview saved to', preview_path)
