"""Launcher for autoTs_PVGIS殘差校正.py

用法:
  python autoTs_PVGIS殘差校正_launcher.py [iterations] [delay_seconds]

參數:
  iterations      重複執行次數。
                  若為 0 或不指定，則無限重複直到按 Ctrl+C 停止。
  delay_seconds   每次執行之間的等待秒數，預設 10 秒。

此啟動器會在同一資料夾下尋找 autoTs_PVGIS殘差校正.py，
並使用當前 Python 直譯器重複執行它。
"""

import sys
import subprocess
import time
from pathlib import Path


DEFAULT_DELAY_SECONDS = 10


def parse_args(argv):
    iterations = 0
    delay = DEFAULT_DELAY_SECONDS
    if len(argv) >= 1:
        try:
            iterations = int(argv[0])
        except ValueError:
            print(f"無效的 iterations: {argv[0]}，請輸入整數。")
            sys.exit(1)
    if len(argv) >= 2:
        try:
            delay = float(argv[1])
        except ValueError:
            print(f"無效的 delay_seconds: {argv[1]}，請輸入數字。")
            sys.exit(1)
    return iterations, delay


def run_script(script_path: Path, iteration: int) -> int:
    cmd = [sys.executable, str(script_path)]
    print(f"[{iteration}] 執行: {' '.join(cmd)}")
    try:
        completed = subprocess.run(cmd, check=False)
    except Exception as exc:
        print(f"[{iteration}] 執行失敗: {exc}")
        return 1
    print(f"[{iteration}] 結束，返回碼: {completed.returncode}")
    return completed.returncode


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    iterations, delay = parse_args(argv)

    launcher_dir = Path(__file__).parent
    script_path = launcher_dir / "autoTs_PVGIS殘差校正.py"
    if not script_path.exists():
        print(f"找不到目標腳本: {script_path}")
        sys.exit(1)

    if iterations <= 0:
        print("啟動無限迴圈執行 autoTs_PVGIS殘差校正.py，按 Ctrl+C 停止。")
    else:
        print(f"啟動 {iterations} 次重複執行，每次間隔 {delay} 秒。")

    count = 0
    try:
        while iterations <= 0 or count < iterations:
            count += 1
            run_script(script_path, count)
            if iterations > 0 and count >= iterations:
                break
            print(f"等待 {delay} 秒後繼續執行...\n")
            time.sleep(delay)
    except KeyboardInterrupt:
        print("\n使用者已中斷執行。")


if __name__ == "__main__":
    main()
