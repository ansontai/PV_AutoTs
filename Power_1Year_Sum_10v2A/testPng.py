# test_plot_generation.py
from pathlib import Path
import pandas as pd
import numpy as np
import re
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available!")
    sys.exit(1)

TARGET_COLUMN = "Wh"

def save_single_column_plot(
    plot_path: Path,
    forecast_index: pd.DatetimeIndex,
    forecast_values: np.ndarray,
    column_name: str,
    title: str = "Forecast",
) -> bool:
    """Save a single-column forecast plot with the same style as forecast_365d_future_*.png"""
    fig, ax = plt.subplots(figsize=(6, 3), dpi=300, constrained_layout=True)
    ax.plot(forecast_index, forecast_values, label=column_name, color="dimgray", linewidth=2.5)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel(column_name)  # 使用欄位名作為 Y 軸標籤
    ax.grid(alpha=0.35, linestyle=":", linewidth=0.8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"✓ Saved: {plot_path.name}")
    return True


def generate_single_line_plots_for_csv(csv_path: Path, output_dir: Path) -> None:
    """Generate single-line plots for each column in the CSV"""
    print(f"\n📊 Processing: {csv_path.name}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"✗ Failed to read CSV: {exc}")
        return

    # 找時間欄
    time_col = None
    for candidate in ("period_start", "date", "period"):
        if candidate in df.columns:
            time_col = candidate
            break
    
    if time_col is None:
        print(f"✗ No time column found in {csv_path.name}")
        return

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.set_index(time_col)
    
    print(f"  欄位: {list(df.columns)}")
    
    # 為每個欄位生成圖片
    for col in df.columns:
        try:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.isna().all():
                print(f"  - {col}: 全是 NaN，跳過")
                continue
            
            safe_col = re.sub(r"[^0-9A-Za-z_.-]", "_", col)
            plot_file = output_dir / f"{csv_path.stem}_{safe_col}.png"
            
            title = f"{col} Forecast"
            save_single_column_plot(
                plot_path=plot_file,
                forecast_index=df.index,
                forecast_values=series.to_numpy(dtype=float, copy=False),
                column_name=col,
                title=title,
            )
        except Exception as exc:
            print(f"  ✗ {col}: {exc}")


if __name__ == "__main__":
    # 測試 CSV 路徑
    csv_path = Path(r"t:\OneDrive\1TB\School\PV_autoTs\Power_1Year_Sum_10v2A\output\default\L6_Lancher-S92b_G50_120d_TmyTrue_1Seed-FitEn\10vL6_autots_365d_forecast_20260513_163950(good_0.686)\forecast_365d_20260513_163950.csv")
    
    # 輸出目錄（放在同一個資料夾）
    output_dir = csv_path.parent
    
    if not csv_path.exists():
        print(f"✗ CSV file not found: {csv_path}")
        sys.exit(1)
    
    generate_single_line_plots_for_csv(csv_path, output_dir)
    print("\n✓ 測試完成！檢查輸出目錄中的 PNG 檔案")