import os
root=r"t:\OneDrive\1TB\School\PV_autoTs\.venv\Lib\site-packages\neuralforecast"
for dirpath,dirnames,filenames in os.walk(root):
    for fn in filenames:
        if fn.endswith('.py'):
            path=os.path.join(dirpath,fn)
            try:
                with open(path,'r',encoding='utf-8',errors='ignore') as f:
                    s=f.read()
            except Exception:
                continue
            if 'distributed' in s or 'pytorch_lightning.utilities' in s or 'rank_zero_only' in s:
                print(path)
                for i,line in enumerate(s.splitlines(),1):
                    if 'distributed' in line or 'pytorch_lightning.utilities' in line or 'rank_zero_only' in line:
                        print(f"  {i}: {line.strip()}")
                print()
