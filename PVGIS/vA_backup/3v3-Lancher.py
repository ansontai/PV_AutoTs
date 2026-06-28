from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).parent
INPUT_DIR = HERE / 'input'
OUTPUT_DIR = HERE / 'output'
RAW_INPUT = INPUT_DIR / 'tmy_24.148_120.703_2005_2023.csv'
FINAL_OUTPUT = OUTPUT_DIR / 'tmy_24.148_120.703_2005_2023[mapping].csv'
BASE_LAUNCHER = HERE / '3v3_map_pvgis_daily_to_solarrecord.py'


def run(cmd):
    print('> ', ' '.join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def main():
    if not RAW_INPUT.exists():
        print('找不到輸入檔:', RAW_INPUT)
        sys.exit(2)

    if not BASE_LAUNCHER.exists():
        print('找不到 3v3 腳本:', BASE_LAUNCHER)
        sys.exit(3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if FINAL_OUTPUT.exists():
        FINAL_OUTPUT.unlink()

    run([sys.executable, str(BASE_LAUNCHER), '--raw', str(RAW_INPUT), '--force'])

    candidates = sorted(
        OUTPUT_DIR.glob('*solarrecord-ready.csv'),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        print('找不到 3v3 產生的 mapping 檔案，請確認流程是否成功。')
        sys.exit(4)

    generated = candidates[0]
    shutil.copy2(generated, FINAL_OUTPUT)
    print('已輸出:', FINAL_OUTPUT)


if __name__ == '__main__':
    main()