#!/usr/bin/env python3
"""
最終演示：使用修改後的主程式邏輯生成單線圖表
"""

from pathlib import Path
import sys
import logging
import pandas as pd
import numpy as np
import re

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# matplotlib 配置
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

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
    """Save a single-column forecast plot (each CSV column as a separate plot).
    
    Style matches forecast_365d_future_*.png: single line, no confidence bounds.
    Uses column name as both Y-axis label and legend label.
    """
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


def generate_plots_for_forecast_365d_csvs(
    out_dir: Path,
    actual_series = None,
    lastvalue_series = None,
    train_series = None,
    logger: logging.Logger = None,
) -> None:
    """Scan for forecast_365d_*.csv and generate per-column single-line plots.
    
    Generates one plot per column (excluding date/time columns).
    Each plot shows only that column's values as a single line.
    Style matches forecast_365d_future_*.png.
    """
    try:
        for csv_path in sorted(out_dir.glob("forecast_365d_*.csv")):
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                if logger:
                    logger.info(f"Failed to read CSV {csv_path}: {exc}")
                continue

            # Try common time column names
            time_col = None
            for candidate in ("period_start", "date", "period"):
                if candidate in df.columns:
                    time_col = candidate
                    break
            if time_col is None:
                if logger:
                    logger.info(f"No time column found in {csv_path}; skipping")
                continue

            try:
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
                df = df.set_index(time_col)
            except Exception as exc:
                if logger:
                    logger.info(f"Failed to set time index for {csv_path}: {exc}")
                continue

            # Generate a plot for each column
            plot_count = 0
            for col in df.columns:
                try:
                    series = pd.to_numeric(df[col], errors="coerce")
                    if series.isna().all():
                        # Skip columns that are entirely NaN
                        continue
                    
                    # Create safe filename from column name
                    safe_col = re.sub(r"[^0-9A-Za-z_.-]", "_", col)
                    plot_file = out_dir / f"{csv_path.stem}_{safe_col}.png"
                    
                    # Generate single-column plot
                    title = f"{col} Forecast"
                    success = save_single_column_forecast_plot(
                        plot_path=plot_file,
                        forecast_index=df.index,
                        forecast_values=series.to_numpy(dtype=float, copy=False),
                        column_label=col,
                        title=title,
                        logger=logger,
                    )
                    if success:
                        plot_count += 1
                        if logger:
                            logger.info(f"✓ {plot_file.name}")
                except Exception as exc:
                    if logger:
                        logger.info(f"✗ {col}: {exc}")
            
            if logger:
                logger.info(f"  完成: 生成 {plot_count} 張圖片\n")
    except Exception as exc:
        if logger:
            logger.info(f"generate_plots_for_forecast_365d_csvs failed: {exc}")


if __name__ == "__main__":
    print("=" * 70)
    print("🎯 最終演示：新的單線圖表生成功能")
    print("=" * 70)
    
    # 測試 CSV 路徑
    csv_path = Path(r"t:\OneDrive\1TB\School\PV_autoTs\Power_1Year_Sum_10v2A\output\default\L6_Lancher-S92b_G50_120d_TmyTrue_1Seed-FitEn\10vL6_autots_365d_forecast_20260513_163950(good_0.686)\forecast_365d_20260513_163950.csv")
    output_dir = csv_path.parent / "demo_output"
    output_dir.mkdir(exist_ok=True)
    
    # 複製 CSV 到演示目錄
    import shutil
    demo_csv = output_dir / csv_path.name
    shutil.copy(csv_path, demo_csv)
    
    print(f"\n📊 測試 CSV: {csv_path.name}")
    print(f"   位置: {output_dir}")
    print(f"   已複製 CSV 到演示目錄\n")
    
    print("🔄 生成單線圖表...\n")
    generate_plots_for_forecast_365d_csvs(
        out_dir=output_dir,
        actual_series=None,
        lastvalue_series=None,
        train_series=None,
        logger=logger,
    )
    
    # 統計結果
    import glob
    png_files = sorted(glob.glob(str(output_dir / "*.png")))
    
    print(f"✅ 演示完成！\n")
    print(f"📁 輸出目錄: {output_dir}")
    print(f"📊 生成圖片: {len(png_files)} 張\n")
    
    print("生成的圖片:")
    for png in png_files:
        size_kb = Path(png).stat().st_size / 1024
        print(f"  • {Path(png).name} ({size_kb:.1f} KB)")
    
    print("\n" + "=" * 70)
    print("✨ 每個欄位都有一張對應的單線圖表")
    print("=" * 70)
