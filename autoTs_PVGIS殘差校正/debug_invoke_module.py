import importlib.util
import traceback
from pathlib import Path

module_path = Path(__file__).parent / 'autoTs_PVGIS殘差校正.py'
spec = importlib.util.spec_from_file_location('mainmod', str(module_path))
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
    print('module imported OK')
    pvgis_file = Path(__file__).parent / 'input' / 'Timeseries_24.148_120.703_E5_0kWp_crystSi_25_35deg_1deg_2005_2005[UTC+8][daily][scaled].csv'
    solar_file = Path(__file__).parent / 'input' / 'SolarRecord(260228)_d_forWh_WithCodis.csv'
    print('calling residual_correct_with_autots...')
    res = mod.residual_correct_with_autots(pvgis_fp=pvgis_file, solar_fp=solar_file, forecast_length=30, autots_params={'max_generations':1,'model_list':'superfast'}, output_csv=Path(__file__).parent / 'debug_residual_corrected_forecast.csv')
    print('call finished. merged_history rows:', len(res['merged_history']))
    print('forecast rows:', len(res['forecast']))
except Exception:
    traceback.print_exc()
    print('failed invoking module')
