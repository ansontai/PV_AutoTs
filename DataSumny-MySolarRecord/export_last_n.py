import pandas as pd
import json
from pathlib import Path

N = 20
p = Path('input/SolarRecord_260310_1829-row.csv')
if not p.exists():
    raise SystemExit(f'input file not found: {p}')

df = pd.read_csv(p, low_memory=False)
# if LocalTime header missing, assume first column is timestamp
if 'LocalTime' not in df.columns:
    cols = ['LocalTime'] + [f'col{i}' for i in range(1, len(df.columns))]
    df.columns = cols

# parse timestamps permissively
df['LocalTime_parsed'] = pd.to_datetime(df['LocalTime'], errors='coerce')
# drop invalid and sort
df = df.dropna(subset=['LocalTime_parsed']).sort_values('LocalTime_parsed')

last = df.tail(N).copy()

outdir = Path('output')
outdir.mkdir(parents=True, exist_ok=True)
csvp = outdir / 'last_20_parsed.csv'
jsonp = outdir / 'last_20_parsed.json'

last.to_csv(csvp, index=False, encoding='utf-8-sig')

records = last.to_dict(orient='records')
for r in records:
    v = r.get('LocalTime_parsed')
    if pd.notna(v):
        # pandas Timestamp -> isoformat
        try:
            r['LocalTime_parsed'] = pd.Timestamp(v).isoformat()
        except Exception:
            r['LocalTime_parsed'] = str(v)

with open(jsonp, 'w', encoding='utf-8-sig') as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f'Saved: {csvp}\nSaved: {jsonp}\n')
print(last[['LocalTime','LocalTime_parsed']].to_string(index=False))
