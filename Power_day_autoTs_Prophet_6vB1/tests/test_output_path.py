import os
import importlib.util

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODULE_PATH = os.path.join(BASE_DIR, '6vB1-autoTs_WeatherToDayWh.py')

spec = importlib.util.spec_from_file_location('mod', MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

out_dir = os.path.join(BASE_DIR, 'tmp_test_output')
output_parent, timestamp, count = mod.get_output_parent(out_dir, timestamp=None, start_count=1, name_hint='MyModel')
print('output_parent:', output_parent)
print('timestamp:', timestamp)
print('count:', count)

# basic checks
expected = os.path.join(out_dir, 'MyModel', timestamp)
print('expected:', expected)
print('matches expected:', os.path.normpath(output_parent) == os.path.normpath(expected))

# list created directories
for root, dirs, files in os.walk(out_dir):
    print('DIR:', root)
    for d in dirs:
        print('  subdir:', d)
    for f in files:
        print('  file:', f)
    break
