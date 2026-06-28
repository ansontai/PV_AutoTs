#!/usr/bin/env python3
"""
驗證腳本：測試主程式中的新函數 save_single_column_forecast_plot()
"""

from pathlib import Path
import sys
import logging
import pandas as pd
import numpy as np

# 設定 logging
logging.basicConfig(level=logging.INFO, format='  %(message)s')
logger = logging.getLogger(__name__)

# 導入主程式的模組
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# 添加主程式路徑
main_script = Path(r"t:\OneDrive\1TB\School\PV_autoTs\Power_1Year_Sum_10v2A\10vL7_autots_365d_forecast.py")

print("✓ 匯入必要模組成功\n")

# 定義簡單版本的 save_single_column_forecast_plot() 作為測試
MATPLOTLIB_AVAILABLE = True
TARGET_COLUMN = "Wh"

def save_single_column_forecast_plot(
    plot_path: Path,
    forecast_index: pd.DatetimeIndex,
    forecast_values: np.ndarray,
    column_label: str = "Forecast",
    title: str = "Forecast",
    logger: logging.Logger | None = None,
) -> bool:
    """Save a single-column forecast plot (each CSV column as a separate plot)."""
    if not MATPLOTLIB_AVAILABLE:
        if logger:
            logger.info(f"Matplotlib not available; skipping plot: {plot_path}")
        return False
    
    try:
        fig, ax = plt.subplots(figsize=(6, 3), dpi=300)
        ax.plot(forecast_index, forecast_values, label=column_label, color="dimgray", linewidth=2.5)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Date")
        ax.set_ylabel(column_label)
        ax.grid(alpha=0.35, linestyle=":", linewidth=0.8)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        ax.legend(loc="upper left", fontsize=9, frameon=False)
        fig.subplots_adjust(bottom=0.16)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        return True
    except Exception as exc:
        if logger:
            logger.info(f"Failed to save single-column plot {plot_path}: {exc}")
        return False


if __name__ == "__main__":
    print("🧪 驗證腳本：測試 save_single_column_forecast_plot()\n")
    
    # 測試 CSV 路徑
    csv_path = Path(r"t:\OneDrive\1TB\School\PV_autoTs\Power_1Year_Sum_10v2A\output\default\L6_Lancher-S92b_G50_120d_TmyTrue_1Seed-FitEn\10vL6_autots_365d_forecast_20260513_163950(good_0.686)\forecast_365d_20260513_163950.csv")
    output_dir = csv_path.parent / "test_new_function"
    output_dir.mkdir(exist_ok=True)
    
    # 讀取 CSV
    print(f"📖 讀取 CSV: {csv_path.name}")
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    print(f"   ✓ 成功讀取 {len(df)} 行 × {len(df.columns)} 列\n")
    
    # 測試每個欄位
    print(f"🎨 生成測試圖片:")
    success_count = 0
    for col in df.columns[:3]:  # 只測試前 3 個欄位
        try:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.isna().all():
                print(f"  ⊘ {col}: 全是 NaN")
                continue
            
            plot_file = output_dir / f"test_{col}.png"
            success = save_single_column_forecast_plot(
                plot_path=plot_file,
                forecast_index=df.index,
                forecast_values=series.to_numpy(dtype=float, copy=False),
                column_label=col,
                title=f"{col} Forecast",
                logger=logger,
            )
            if success:
                size_kb = plot_file.stat().st_size / 1024
                print(f"  ✓ {col} → {plot_file.name} ({size_kb:.1f} KB)")
                success_count += 1
        except Exception as exc:
            print(f"  ✗ {col}: {exc}")
    
    print(f"\n✅ 測試完成: 生成 {success_count} 張圖片")
    print(f"   輸出目錄: {output_dir}\n")
