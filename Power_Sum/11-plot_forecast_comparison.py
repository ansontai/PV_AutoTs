from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
TRAIN_CSV_PATH = BASE_DIR / "SolarRecord(260204)_d_forWh_WithCodis.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "forecast_Wh_20260301_20270228_autots_template90d_futureReg.csv"
FORECAST_START = pd.Timestamp("2026-03-01")
FORECAST_END = pd.Timestamp("2027-02-28")


def plot_forecast_comparison(plot_path, index, y_true, y_pred, y_naive, title=None, figsize=(12, 6), dpi=150):
    """Save a simple Forecast vs Actual vs Naive-Lag1 line chart.

    Parameters passed in to avoid reliance on outer scope.
    """
    import matplotlib.dates as mdates
    plt.figure(figsize=figsize)
    # plt.plot(index, y_true, label='Actual', linewidth=2)
    plt.plot(index, y_pred, label='AutoTS Forecast', linewidth=2)
    # plt.plot(index, y_naive, label='Naive Lag-1', linewidth=2, linestyle='--')
    plt.title(title or 'Wh Forecast vs Actual vs Naive Lag-1')
    plt.xlabel('Date')
    plt.ylabel('Wh')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=dpi)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # prefer filled forecast if available
    filled_path = OUTPUT_FILE.with_name(OUTPUT_FILE.stem + "_filled.csv")
    forecast_path = filled_path if filled_path.exists() else OUTPUT_FILE
    if not forecast_path.exists():
        raise FileNotFoundError(f"Forecast CSV not found: {forecast_path}")

    fc = pd.read_csv(forecast_path)
    if 'Date' not in fc.columns:
        raise ValueError('Forecast CSV must include a Date column')
    fc['Date'] = pd.to_datetime(fc['Date'], errors='coerce')
    fc = fc.dropna(subset=['Date']).sort_values('Date')

    # ensure forecast column name
    if 'Wh_pred' not in fc.columns:
        if 'Wh' in fc.columns:
            fc = fc.rename(columns={'Wh': 'Wh_pred'})
        else:
            # try first numeric column
            num_cols = [c for c in fc.columns if c != 'Date']
            if not num_cols:
                raise ValueError('Forecast CSV has no numeric columns')
            fc = fc.rename(columns={num_cols[0]: 'Wh_pred'})

    # read training actuals if available
    if not TRAIN_CSV_PATH.exists():
        train = pd.DataFrame(columns=['Date', 'Wh'])
    else:
        train = pd.read_csv(TRAIN_CSV_PATH)
        date_col = 'Date' if 'Date' in train.columns else ('LocalTime' if 'LocalTime' in train.columns else None)
        if date_col is None or 'Wh' not in train.columns:
            train = pd.DataFrame(columns=['Date', 'Wh'])
        else:
            train['Date'] = pd.to_datetime(train[date_col], errors='coerce')
            train = train.dropna(subset=['Date'])[['Date', 'Wh']]
            train['Wh'] = pd.to_numeric(train['Wh'], errors='coerce')

    forecast_dates = pd.date_range(FORECAST_START, FORECAST_END, freq='D')
    fc = fc.set_index('Date').reindex(forecast_dates).reset_index().rename(columns={'index': 'Date'})

    # merge to get any true values for the forecast window
    merged = fc.merge(train, on='Date', how='left')
    y_pred = merged['Wh_pred'].astype(float).values
    y_true = merged['Wh'].astype(float).values if 'Wh' in merged.columns else np.array([np.nan] * len(merged))

    # compute naive lag-1: first value = last historical Wh, then use previous actual if present else previous naive
    last_hist = None
    if not train.empty:
        last_hist_vals = train[train['Date'] < FORECAST_START]['Wh'].dropna()
        if not last_hist_vals.empty:
            last_hist = float(last_hist_vals.iloc[-1])

    naive = []
    prev = last_hist if last_hist is not None else np.nan
    for i in range(len(merged)):
        naive.append(prev)
        if not np.isnan(y_true[i]):
            prev = float(y_true[i])

    y_naive = np.array(naive, dtype=float)

    plot_path = OUTPUT_DIR / 'forecast_Wh_20260301_20270228_comparison.png'
    plot_forecast_comparison(plot_path, forecast_dates, y_true, y_pred, y_naive,
                             title='Wh: AutoTS vs Actual vs Naive (2026-03-01 → 2027-02-28)')

    print(f'Wrote plot: {plot_path}')


if __name__ == '__main__':
    main()
