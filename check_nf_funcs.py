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
            if 'def forecast(' in s or 'def predict(' in s:
                print(path)
                for i,line in enumerate(s.splitlines(),1):
                    if 'def forecast(' in line or 'def predict(' in line:
                        print(f"  {i}: {line.strip()}")
                print()
