import numpy as np
path=r'T:\OneDrive\1TB\School\python_local\csv\2020-01--2026-02.csv'
df=pd.read_csv(path, parse_dates=['Date'])
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print('numeric columns count', len(numeric_cols))
print(numeric_cols)
for c in numeric_cols:
    print(c, df[c].dtype, df[c].isna().sum())
