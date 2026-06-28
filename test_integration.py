#!/usr/bin/env python3
"""
集成測試：使用主程式中的新函數生成單線圖表
"""

from pathlib import Path
import sys
import logging

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 從主程式導入相關函數
sys.path.insert(0, str(Path(__file__).parent))

try:
    from Power_1Year_Sum_10v2A.test_plot_generation import (
        generate_single_line_plots_for_csv,
    )
    print("✓ 成功導入測試函數")
except ImportError as e:
    print(f"❌ 無法導入: {e}")
    sys.exit(1)


if __name__ == "__main__":
    print("\n🔧 集成測試：驗證主程式新函數\n")
    
    # 測試 CSV 路徑
    csv_path = Path(r"t:\OneDrive\1TB\School\PV_autoTs\Power_1Year_Sum_10v2A\output\default\L6_Lancher-S92b_G50_120d_TmyTrue_1Seed-FitEn\10vL6_autots_365d_forecast_20260513_163950(good_0.686)\forecast_365d_20260513_163950.csv")
    output_dir = csv_path.parent
    
    if not csv_path.exists():
        print(f"❌ CSV 檔案不存在")
        sys.exit(1)
    
    # 使用測試函數
    print(f"📊 測試 CSV: {csv_path.name}\n")
    generate_single_line_plots_for_csv(csv_path, output_dir)
    
    # 驗證輸出
    print("\n✅ 集成測試完成\n")
    
    # 列出生成的圖片
    import glob
    png_files = sorted(glob.glob(str(output_dir / "forecast_365d_20260513_163950_*.png")))
    print(f"生成的圖片 ({len(png_files)} 張):")
    for png in png_files:
        size_kb = Path(png).stat().st_size / 1024
        print(f"  ✓ {Path(png).name} ({size_kb:.1f} KB)")
