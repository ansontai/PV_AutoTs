import json
from pathlib import Path
raw_mw = '{"MAE":5,"SMAPE":1,"RMSE":0}'
print('raw_mw:', raw_mw)
p = Path(str(raw_mw))
print('p.exists()', p.exists())
try:
    mw = json.loads(str(raw_mw))
    print('mw', mw)
except Exception as e:
    print('json.loads failed', e)


def _normalize_mw(d):
    nd = {}
    for k, v in (d.items() if isinstance(d, dict) else []):
        kk = str(k)
        if kk.lower().endswith('_weighting'):
            nd[kk.lower()] = v
        else:
            ku = kk.upper()
            if ku == 'MAE':
                nd['mae_weighting'] = v
            elif ku == 'SMAPE':
                nd['smape_weighting'] = v
            elif ku == 'RMSE':
                nd['rmse_weighting'] = v
            elif ku == 'MASE':
                nd['mase_weighting'] = v
            else:
                nd[kk.lower()] = v
    try:
        others = [float(x) for kk, x in nd.items() if kk != 'mae_weighting' and isinstance(x, (int, float))]
        max_other = max(others) if others else 0.0
    except Exception:
        max_other = 0.0
    nd['mae_weighting'] = max(float(nd.get('mae_weighting', 0)), max_other + 1)
    return nd

print('normalized:', json.dumps(_normalize_mw(mw), indent=2))
