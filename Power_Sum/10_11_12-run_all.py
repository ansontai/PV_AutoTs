from pathlib import Path
import subprocess
import sys
import time

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    "10-autoTs_Wh_20260301_20270228_with_template_and_future_reg.py",
    "10b-fill_missing_forecast.py",
    "11-plot_forecast_comparison.py",
    "12-sum_forecast_wh.py",
]


def run_script(script_name: str):
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        return False, f"Script not found: {script_path}"

    log_path = OUTPUT_DIR / f"run_{script_name.replace('.py','')}.log"
    cmd = [sys.executable, str(script_path)]
    start = time.time()
    proc = subprocess.run(
      cmd, 
      # capture_output=True, 
      capture_output=False,
      text=True
      )
    elapsed = time.time() - start

    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"Command: {' '.join(cmd)}\n")
        fh.write(f"Returncode: {proc.returncode}\n")
        fh.write(f"Elapsed: {elapsed:.2f}s\n\n")
        fh.write("STDOUT\n" + "=" * 40 + "\n")
        fh.write(proc.stdout or "")
        fh.write("\n\nSTDERR\n" + "=" * 40 + "\n")
        fh.write(proc.stderr or "")

    return proc.returncode == 0, str(log_path)


def main():
    any_failed = False
    for script in SCRIPTS:
        print(f"Running {script} ...")
        ok, info = run_script(script)
        if ok:
            print(f"OK — log: {info}")
        else:
            print(f"FAILED — {info}")
            any_failed = True

    if any_failed:
        print("One or more scripts failed. See logs in:", OUTPUT_DIR)
        sys.exit(2)

    print("All scripts finished successfully.")


if __name__ == '__main__':
    main()
