from pathlib import Path
import pandas as pd

base = Path(__file__).parent / "input"
pvgis_file = base / "Timeseries_24.148_120.703_E5_0kWp_crystSi_25_35deg_1deg_2005_2005[UTC+8][daily][scaled].csv"
solar_file = base / "SolarRecord(260228)_d_forWh_WithCodis.csv"

print('PVGIS exists:', pvgis_file.exists(), pvgis_file)
print('Solar exists:', solar_file.exists(), solar_file)

pvgis = pd.read_csv(pvgis_file, parse_dates=['date'])
solar = pd.read_csv(solar_file, parse_dates=['LocalTime'])

pvgis['date'] = pd.to_datetime(pvgis['date']).dt.normalize()
solar['date'] = pd.to_datetime(solar['LocalTime']).dt.normalize()

print('\nPVGIS shape:', pvgis.shape)
print('PVGIS date range:', pvgis['date'].min(), '->', pvgis['date'].max())
print('\nSolar shape:', solar.shape)
print('Solar date range:', solar['date'].min(), '->', solar['date'].max())

pv_dates = set(pvgis['date'].dt.date.unique())
so_dates = set(solar['date'].dt.date.unique())
print('\nOverlap count (unique dates):', len(pv_dates & so_dates))

# day-of-year mapping check
tmp = pvgis.copy()
tmp['doy'] = tmp['date'].dt.dayofyear
if 'P_mapped_Wh' not in tmp.columns:
    print('\nPVGIS missing P_mapped_Wh column!')
else:
    doy_map = tmp.groupby('doy')['P_mapped_Wh'].mean()
    solar_dates = pd.to_datetime(solar['date']).dt.normalize()
    mapped = pd.Series(pd.DatetimeIndex(solar_dates).dayofyear).map(doy_map).values
    mapped = pd.Series(mapped, index=solar_dates)
    print('\nMapped NaNs:', mapped.isna().sum(), '/', len(mapped))

    solar2 = solar.rename(columns={'LocalTime':'date','Wh':'y_obs'})
    solar2['date'] = pd.to_datetime(solar2['date']).dt.normalize()
    solar2['y_pvgis_daily'] = mapped.values
    both = solar2.dropna(subset=['y_obs','y_pvgis_daily'])
    print('Rows with both obs and mapped PVGIS:', len(both))
    if len(both) > 0:
        print('\nSample merged rows:')
        print(both[['date','y_obs','y_pvgis_daily']].head(10).to_string(index=False))

print('\nDone diagnostics.')
