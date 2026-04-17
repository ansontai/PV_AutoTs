import pandas as pd
import os
path = r'T:\OneDrive\1TB\School\python_local\Power_day_v4-autoTs_vs_PVGIS\input\Timeseries_24.148_120.703_E5_0kWp_crystSi_25_35deg_1deg_2005_2005[UTC+8][daily][scaled].csv'
print('path', path)
print('exists', os.path.exists(path))
if os.path.exists(path):
    df = pd.read_csv(path, nrows=5)
    print('columns', list(df.columns))
    print(df.head(2).to_string(index=False))
