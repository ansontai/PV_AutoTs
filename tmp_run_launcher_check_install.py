import runpy
import subprocess
import os
import sys
import traceback

orig_run = subprocess.run

def mock_run(cmd, *a, **kw):
    try:
        print("MOCK_SUBPROCESS_RUN:", cmd)
    except Exception as e:
        print("MOCK_SUBPROCESS_RUN: (print error)", e)
    return 0

subprocess.run = mock_run

script_path = os.path.join(os.path.dirname(__file__), 'Power_day_autoTs_Prophet_6vB1', '6vB1_Lancher-pMLP-30Seed.py')
print('Wrapper: running', script_path)
try:
    runpy.run_path(script_path, run_name='__main__')
except SystemExit as e:
    print('Script exited with SystemExit:', e)
except Exception:
    traceback.print_exc()
finally:
    subprocess.run = orig_run
