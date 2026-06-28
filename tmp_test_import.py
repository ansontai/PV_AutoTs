import importlib.util, os
p = r't:/OneDrive/1TB/School/PV_autoTs/Power_day_autoTs_Prophet_6vB1/6vB1-autoTs_WeatherToDayWh.py'
spec = importlib.util.spec_from_file_location('childmod', p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
get_output_parent = mod.get_output_parent
out, ts, cnt = get_output_parent(r't:/OneDrive/1TB/School/PV_autoTs/Power_day_autoTs_Prophet_6vB1/test_output_root', name_hint='test_model', user_tag='mytag')
print('CREATED:', out)
print('TIMESTAMP:', ts)
print('COUNT:', cnt)
print('EXISTS:', os.path.exists(out))
