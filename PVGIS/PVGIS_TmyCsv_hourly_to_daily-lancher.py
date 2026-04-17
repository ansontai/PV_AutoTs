"""Launcher for PVGIS_TmyCsv_handle.py (方案 A)

用法:
  - 直接執行: python PVGIS/launch_pvgis_tmy.py
  - 覆寫輸入/輸出: python PVGIS/launch_pvgis_tmy.py <input.csv> <output.csv>

預設會使用:
  t:\\raw\\tmy_24.148_120.703_2005_2023.csv
  t:\\output\\tmy_24.148_120.703_2005_2023[daily].csv
"""
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent
DEFAULT_IN = str(BASE_DIR / "raw" / "tmy_24.148_120.703_2005_2023.csv")
DEFAULT_OUT = str(BASE_DIR / "output" / "tmy_24.148_120.703_2005_2023[UTC+8][daily].csv")


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) >= 2:
        inp = argv[0]
        outp = argv[1]
    elif len(argv) == 1:
        inp = argv[0]
        outp = DEFAULT_OUT
    else:
        inp = DEFAULT_IN
        outp = DEFAULT_OUT

    script_path = Path(__file__).parent / "PVGIS_TmyCsv_hourly_to_daily.py"
    cmd = [sys.executable, str(script_path), inp, outp]

    print("執行：", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("執行失敗，返回碼:", e.returncode)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
