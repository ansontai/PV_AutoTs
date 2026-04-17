from pathlib import Path
p = Path(r't:/OneDrive/1TB/School/python_local/Power_day_v3/6v3-autoTs_WeatherToDayWh.py')
text = p.read_text(encoding='utf-8')
lines = text.splitlines()
new_lines = []
skip = False
for line in lines:
    if skip:
        if line.strip() == ']':
            skip = False
        continue
    if line.strip().startswith('default_model_list = ['):
        new_lines.append("default_model_list = ['ARIMA']")
        if line.strip().endswith(']'):
            continue
        skip = True
        continue
    if line.strip().startswith('default_ensemble ='):
        new_lines.append("default_ensemble = ['simple']")
        continue
    if line.strip().startswith('default_n_jobs ='):
        new_lines.append('default_n_jobs = 1')
        continue
    if line.strip().startswith('default_transformer_list ='):
        new_lines.append('default_transformer_list = []')
        continue
    new_lines.append(line)
p.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
print('updated')
