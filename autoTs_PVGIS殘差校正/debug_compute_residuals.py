from pathlib import Path
import pandas as pd
import numpy as np

base = Path(__file__).parent / "input"
pvgis_file = base / "Timeseries_24.148_120.703_E5_0kWp_crystSi_25_35deg_1deg_2005_2005[UTC+8][daily][scaled].csv"
solar_file = base / "SolarRecord(260228)_d_forWh_WithCodis.csv"

pvgis = pd.read_csv(pvgis_file, parse_dates=['date'])
solar = pd.read_csv(solar_file, parse_dates=['LocalTime'])

pvgis['date'] = pd.to_datetime(pvgis['date']).dt.normalize()
solar['date'] = pd.to_datetime(solar['LocalTime']).dt.normalize()

# map_pvgis_to_dates

def map_pvgis_to_dates(pvgis_df, target_dates):
    target_dates = pd.to_datetime(target_dates)
    min_t, max_t = target_dates.min(), target_dates.max()
    mask = (pvgis_df['date'] >= min_t) & (pvgis_df['date'] <= max_t)
    if mask.any():
        sub = pvgis_df.loc[mask, ['date', 'P_mapped_Wh']]
        mapped = pd.DataFrame({'date': target_dates}).merge(sub, on='date', how='left')['P_mapped_Wh']
        mapped.index = target_dates
        return mapped
    tmp = pvgis_df.copy()
    tmp['doy'] = tmp['date'].dt.dayofyear
    doy_map = tmp.groupby('doy')['P_mapped_Wh'].mean()
    doy = pd.DatetimeIndex(target_dates).dayofyear
    mapped_vals = pd.Series(doy).map(doy_map).values
    mapped = pd.Series(mapped_vals, index=target_dates)
    if mapped.isna().any():
        month_map = tmp.groupby(tmp['date'].dt.month)['P_mapped_Wh'].mean()
        months = pd.DatetimeIndex(target_dates).month
        mapped = mapped.fillna(pd.Series(months, index=target_dates).map(month_map))
    return mapped

# Prepare obs
obs = solar[['date', 'Wh']].rename(columns={'Wh':'y_obs'}).copy()
obs_dates = obs['date']
obs['y_pvgis_daily'] = map_pvgis_to_dates(pvgis, obs_dates).values

print('obs rows:', len(obs))
print('y_obs NaNs:', obs['y_obs'].isna().sum())
print('y_pvgis_daily NaNs:', pd.isna(obs['y_pvgis_daily']).sum())

train = obs.dropna(subset=['y_obs','y_pvgis_daily']).copy()
print('train rows (both present):', len(train))
if len(train)>0:
    train['month'] = train['date'].dt.month
    pvgis_sum = train.groupby('month')['y_pvgis_daily'].sum().replace(0,1e-6)
    obs_sum = train.groupby('month')['y_obs'].sum()
    k = (obs_sum / pvgis_sum).reindex(range(1,13)).fillna(1.0)
    print('k sample:', k.dropna().head().to_dict())
    obs['month'] = obs['date'].dt.month
    obs['y_pvgis_scaled'] = obs['month'].map(k) * obs['y_pvgis_daily']
    obs['residual'] = obs['y_obs'] - obs['y_pvgis_scaled']
    print('residual NaNs:', obs['residual'].isna().sum())
    print('\nresidual sample:')
    print(obs[['date','y_obs','y_pvgis_daily','y_pvgis_scaled','residual']].head(10).to_string(index=False))
else:
    print('No training rows; cannot compute k')
