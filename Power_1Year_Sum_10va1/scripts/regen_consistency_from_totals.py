from pathlib import Path
import importlib.util
import logging
import sys

# locate module file
base_dir = Path(__file__).resolve().parent.parent
module_file = base_dir / "10vL3_autots_365d_forecast.py"

spec = importlib.util.spec_from_file_location("autots_mod", str(module_file))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# set up logger
logger = logging.getLogger("regen_consistency")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(handler)

# paths (adjust timestamp if needed)
run_dir = base_dir / "output" / "10vL2_autots_365d_forecast_20260513_100353"
totals_csv = run_dir / "forecast_365d_totals_20260513_100353.csv"
output_dir = run_dir
timestamp = "20260513_100353"

print(f"Using totals: {totals_csv}")
res = mod.compute_and_save_annual_consistency(totals_csv, output_dir, timestamp, logger)
print("Created consistency file:", res)
