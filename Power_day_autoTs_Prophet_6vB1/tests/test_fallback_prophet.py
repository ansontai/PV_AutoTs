import pandas as pd
import numpy as np
import importlib.util
import pathlib


# load module by path because filename contains '-' and isn't a valid module name
here = pathlib.Path(__file__).resolve().parent.parent
mod_path = here / '6v3b-autoTs_WeatherToDayWh.py'
spec = importlib.util.spec_from_file_location('autots_main', str(mod_path))
autots_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(autots_main)
fallback_prophet_predict = autots_main.fallback_prophet_predict


class FakeProphet:
    def __init__(self):
        self.history = None

    def fit(self, df):
        # store history used by make_future_dataframe
        self.history = df.copy()

    def make_future_dataframe(self, periods, freq):
        # create a date range from history start to last + periods
        last = pd.to_datetime(self.history['ds']).max()
        start = pd.to_datetime(self.history['ds']).min()
        # include history length + periods
        total = len(self.history) + periods
        idx = pd.date_range(start=start, periods=total, freq=freq)
        return pd.DataFrame({'ds': idx})

    def predict(self, future):
        # return a predictable yhat sequence
        yhat = np.arange(len(future)) * 1.0
        return pd.DataFrame({'ds': future['ds'], 'yhat': yhat})


def test_fallback_prophet_predict_basic():
    # build minimal training df
    dates = pd.date_range(start='2020-01-01', periods=5, freq='D')
    y = np.array([10, 12, 11, 13, 12], dtype=float)
    dfp_train = pd.DataFrame({'ds': dates, 'y': y})

    horizon = 3
    result = fallback_prophet_predict(dfp_train, horizon, FakeProphet)

    # result should have length == horizon and column 'Wh'
    assert isinstance(result, pd.DataFrame)
    assert 'Wh' in result.columns
    assert len(result) == horizon
    # index should be DatetimeIndex
    assert isinstance(result.index, pd.DatetimeIndex)
