import traceback, sys

def tryimp(name):
    try:
        m = __import__(name)
        print('OK', name, getattr(m, '__file__', None))
    except Exception as e:
        print('ERR', name, e)

tryimp('PIL')
try:
    from PIL import _imaging
    print('_imaging OK', getattr(_imaging, '__file__', None))
except Exception as e:
    print('_imaging ERR', e)

tryimp('PIL.Image')
tryimp('matplotlib')
try:
    import importlib
    importlib.import_module('matplotlib._c_internal_utils')
    print('matplotlib._c_internal_utils OK')
except Exception as e:
    print('matplotlib._c_internal_utils ERR', e)

try:
    import statsmodels.regression
    print('statsmodels.regression OK')
except Exception as e:
    print('statsmodels.regression ERR', e)

try:
    import statsmodels.tsa.filters.hp_filter
    print('statsmodels.tsa.filters.hp_filter OK')
except Exception as e:
    print('statsmodels.tsa.filters.hp_filter ERR', e)

print('Import test done')
