import pandas as pd

a = pd.read_csv("Power_1Year_Sum_V2/input/tmy_24.148_120.703_2005_2023[UTC+8][daily].csv", nrows=0).columns.tolist()
b = pd.read_csv("Power_1Year_Sum_V2/input/SolarRecord(260228)_d_forWh_WithCodis-date.csv", nrows=0).columns.tolist()

set_a = set(a)
set_b = set(b)

only_a = sorted(set_a - set_b)
only_b = sorted(set_b - set_a)
common = sorted(set_a & set_b)

print("Only in A:", only_a)
print("Only in B:", only_b)
print("Common:", common)