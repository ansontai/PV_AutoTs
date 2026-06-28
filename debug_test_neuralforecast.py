import pandas as pd
from autots.models.neural_forecast import NeuralForecast

# minimal panel with two series
idx = pd.date_range('2023-01-01', periods=10, freq='D')
df = pd.DataFrame({'s1': range(10), 's2': range(10,20)}, index=idx)

nf = NeuralForecast(forecast_length=2)
try:
    nf.fit(df)
    print('fit completed')
except Exception as e:
    import traceback
    traceback.print_exc()
    print('exception:', e)
