import os
import shutil
import math
import numpy as np
import pandas as pd

# Defaults (集中於檔案頂端方便修改)
DEFAULT_INPUT = os.path.join('csv', '2000--202602-d-forWh.csv')
DEFAULT_OUTPUT_SUFFIX = '_4b'  # 會變成 <原檔名> + DEFAULT_OUTPUT_SUFFIX + <副檔名>


def add_features(df, date_col='Date', harmonics=(1, 2, 3), drop_first_month=True, year_days=365.0):
    dates = pd.to_datetime(df[date_col], errors='coerce')
    doy = dates.dt.dayofyear.fillna(0).astype(float)

    for k in harmonics:
        angles = 2 * math.pi * (doy * k / year_days)
        df[f'sin_DOY_k{k}'] = np.sin(angles)
        df[f'cos_DOY_k{k}'] = np.cos(angles)

    month = dates.dt.month.fillna(0).astype(int)
    month_dummies = pd.get_dummies(month, prefix='month')
    if drop_first_month and 'month_1' in month_dummies.columns:
        month_dummies = month_dummies.drop('month_1', axis=1)
    df = pd.concat([df, month_dummies], axis=1)

    def month_to_season(m):
        if m in (3, 4, 5):
            return 'spring'
        if m in (6, 7, 8):
            return 'summer'
        if m in (9, 10, 11):
            return 'autumn'
        return 'winter'

    season_series = month.apply(month_to_season)
    season_dummies = pd.get_dummies(season_series, prefix='season')
    if 'season_winter' in season_dummies.columns:
        season_dummies = season_dummies.drop('season_winter', axis=1)
    df = pd.concat([df, season_dummies], axis=1)

    weekday = dates.dt.weekday
    df['is_weekend'] = (weekday >= 5).astype(int)

    return df


def main():
    src = DEFAULT_INPUT
    if not os.path.exists(src):
        print('Source file not found:', src)
        return

    # backup original if not already backed up
    bak = src + '.bak'
    if not os.path.exists(bak):
        shutil.copy2(src, bak)

    df = pd.read_csv(src)

    if 'Date' in df.columns:
        date_col = 'Date'
    elif 'LocalTime' in df.columns:
        date_col = 'LocalTime'
    else:
        raise RuntimeError('No Date or LocalTime column found')

    df = add_features(df, date_col=date_col, harmonics=(1, 2, 3), drop_first_month=True, year_days=365.0)

    # Output filename: original name + _4b (preserve extension)
    base, ext = os.path.splitext(src)
    out = f"{base}_4b{ext}"

    tmp = out + '.tmp'
    df.to_csv(tmp, index=False)
    shutil.move(tmp, out)
    print('Updated', out, '(backup at', bak, ')')


if __name__ == '__main__':
    main()
