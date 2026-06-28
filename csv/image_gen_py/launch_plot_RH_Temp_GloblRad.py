"""啟動器：呼叫 `plot_RH_Temp_GloblRad.py` 並將輸出放到輸入 CSV 的同一目錄。

用法範例：
    python launch_plot_RH_Temp_GloblRad.py --input "..\SolarRecord(260228)_d_forWh_WithCodis.csv"
    python launch_plot_RH_Temp_GloblRad.py --input "T:\\OneDrive\\1TB\\School\\PV_autoTs\\csv\\SolarRecord(260228)_d_forWh_WithCodis.csv" --output-name "我的圖.png"
"""
from pathlib import Path
import subprocess
import sys
import argparse


DEFAULT_CSV = Path(__file__).parent.parent / "SolarRecord(260228)_d_forWh_WithCodis.csv"
SCRIPT = Path(__file__).parent / "plot_RH_Temp_GloblRad.py"


def main():
    p = argparse.ArgumentParser(description="啟動 plot_RH_Temp_GloblRad，並把輸出存到輸入 CSV 同目錄")
    p.add_argument("--input", "-i", default=str(DEFAULT_CSV), help="輸入 CSV 路徑（預設為 repo csv 下的 SolarRecord...）")
    p.add_argument("--output-name", "-n", default="氣象線條圖.png", help="輸出檔名（只給檔名，啟動器會放到輸入的同一目錄）")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: 找不到輸入 CSV: {input_path}")
        sys.exit(2)

    out_path = input_path.parent / args.output_name

    if not SCRIPT.exists():
        print(f"Error: plotting script not found: {SCRIPT}")
        sys.exit(2)

    # First: generate combined 3-panel plot
    cmd_combined = [sys.executable, str(SCRIPT), "--input", str(input_path), "--output", str(out_path), "--single"]
    print("Running combined plot:", " ".join(cmd_combined))
    try:
        subprocess.run(cmd_combined, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Combined plot failed with exit {e.returncode}")
        sys.exit(e.returncode)

    # Second: generate separate single-column plots (will use the same output stem)
    cmd_split = [sys.executable, str(SCRIPT), "--input", str(input_path), "--output", str(out_path)]
    print("Running split plots:", " ".join(cmd_split))
    try:
        subprocess.run(cmd_split, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Split plots failed with exit {e.returncode}")
        sys.exit(e.returncode)

    print(f"Combined output saved to: {out_path}")
    print(f"Individual files saved with prefix: {out_path.stem}_* in {out_path.parent}")


if __name__ == "__main__":
    main()
