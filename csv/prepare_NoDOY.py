import os
import pandas as pd

def main():
    src = os.path.join('csv', '2000--202602-d-forWh.csv')
    out = os.path.join('csv', '2000--202602-d-forWh-NoDOY.csv')
    if not os.path.exists(src):
        print('Source not found:', src)
        return
    df = pd.read_csv(src)
    if 'day_of_year' in df.columns:
        df = df.drop(columns=['day_of_year'])
    df.to_csv(out, index=False)
    print('Wrote', out)

if __name__ == '__main__':
    main()
