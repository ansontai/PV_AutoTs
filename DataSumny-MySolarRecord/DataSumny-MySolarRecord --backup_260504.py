#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataSumny-MySolarRecord.py
分析 SolarRecord CSV 的 localTime 時間完整性。

功能：
  - 檢查指定 CSV 中 localTime 欄位的缺失情況
  - 統計分鐘/小時/天三種粒度的缺失數與百分比（雙分母）
  - 輸出終端摘要與 JSON 報告

用法：
  python DataSumny-MySolarRecord.py [--input <csv_file>] [--output <dir>]

預設：
  輸入：DataSumny-MySolarRecord/input/SolarRecord_260310_1829-row.csv
  輸出：DataSumny-MySolarRecord/output/
"""

import argparse
import sys
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


class TimeCompletenessAnalyzer:
    """時間完整性分析器"""
    
    def __init__(self, csv_path: str, output_dir: str = None):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir) if output_dir else self.csv_path.parent.parent / "output"
        self.df = None
        self.start_time = None
        self.end_time = None
        self.data_points = 0
        self.data_points_cleaned = 0
        self.results = {}
        
    def load_and_clean_data(self):
        """讀取 CSV 並清理時間欄位"""
        print(f"[INFO] 讀取檔案: {self.csv_path}")
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"檔案不存在: {self.csv_path}")
        
        try:
            self.df = pd.read_csv(self.csv_path, low_memory=False)
        except Exception as e:
            raise RuntimeError(f"讀取 CSV 失敗: {e}")
        
        # 檢查 localTime 欄位
        if 'LocalTime' not in self.df.columns:
            raise ValueError(f"缺少 'LocalTime' 欄位。現有欄位: {list(self.df.columns)}")
        
        self.data_points = len(self.df)
        print(f"[INFO] 原始資料筆數: {self.data_points}")
        
        # 轉換為 datetime，允許各種常見格式（含秒或不含秒），無效的日期轉為 NaT
        # 不指定 format，讓 pandas 自行推斷
        self.df['LocalTime'] = pd.to_datetime(self.df['LocalTime'], errors='coerce')
        
        # 移除無效日期
        self.df = self.df.dropna(subset=['LocalTime'])
        self.data_points_cleaned = len(self.df)
        
        if self.data_points_cleaned == 0:
            raise ValueError("所有時間戳都無效或缺失")
        
        print(f"[INFO] 清理後筆數: {self.data_points_cleaned} (移除 {self.data_points - self.data_points_cleaned} 筆無效記錄)")
        
        # 排序並去重
        self.df = self.df.sort_values('LocalTime').drop_duplicates(subset=['LocalTime'], keep='last')
        self.data_points_cleaned = len(self.df)
        print(f"[INFO] 去重後筆數: {self.data_points_cleaned}")
        
        self.start_time = self.df['LocalTime'].min()
        self.end_time = self.df['LocalTime'].max()
        print(f"[INFO] 時間範圍: {self.start_time} 至 {self.end_time}")
        
    def analyze_minutes(self):
        """分析分鐘粒度的缺失"""
        print("\n[INFO] 分析分鐘粒度缺失...")
        
        # 計算分鐘數
        start_min = self.start_time.floor('min')
        end_min = self.end_time.floor('min')
        minutes_count = int((end_min - start_min).total_seconds() / 60) + 1
        
        # 建立完整分鐘索引
        minute_freq_index = pd.date_range(start=start_min, periods=minutes_count, freq=pd.Timedelta(minutes=1))
        
        expected_count = len(minute_freq_index)
        
        # 對實際時間進行分鐘對齁
        actual_minutes = self.df['LocalTime'].dt.floor('min').unique()
        observed_count = len(actual_minutes)
        
        missing_count = expected_count - observed_count
        pct_expected = (missing_count / expected_count * 100) if expected_count > 0 else 0
        pct_observed = (missing_count / observed_count * 100) if observed_count > 0 else None
        
        result = {
            'expected': expected_count,
            'observed': observed_count,
            'missing': missing_count,
            'pct_vs_expected': round(pct_expected, 2),
            'pct_vs_observed': round(pct_observed, 2) if pct_observed is not None else None,
            'duration_span': str(timedelta(minutes=expected_count - 1))
        }
        
        self.results['minute'] = result
        print(f"  預期筆數: {expected_count}")
        print(f"  觀測筆數: {observed_count}")
        print(f"  缺失筆數: {missing_count}")
        print(f"  缺失比例 (相對預期): {pct_expected:.2f}%")
        if pct_observed is not None:
            print(f"  缺失比例 (相對觀測): {pct_observed:.2f}%")
        else:
            print(f"  缺失比例 (相對觀測): N/A")
        
    def analyze_hours(self):
        """分析小時粒度的缺失"""
        print("\n[INFO] 分析小時粒度缺失...")
        
        # 計算小時數
        start_hour = self.start_time.floor('h')
        end_hour = self.end_time.floor('h')
        hours_count = int((end_hour - start_hour).total_seconds() / 3600) + 1
        
        # 建立完整小時索引
        hour_freq_index = pd.date_range(start=start_hour, periods=hours_count, freq=pd.Timedelta(hours=1))
        
        expected_count = len(hour_freq_index)
        
        # 對實際時間進行小時對齁
        actual_hours = self.df['LocalTime'].dt.floor('h').unique()
        observed_count = len(actual_hours)
        
        missing_count = expected_count - observed_count
        pct_expected = (missing_count / expected_count * 100) if expected_count > 0 else 0
        pct_observed = (missing_count / observed_count * 100) if observed_count > 0 else None
        
        result = {
            'expected': expected_count,
            'observed': observed_count,
            'missing': missing_count,
            'pct_vs_expected': round(pct_expected, 2),
            'pct_vs_observed': round(pct_observed, 2) if pct_observed is not None else None,
            'duration_span': str(timedelta(hours=expected_count - 1))
        }
        
        self.results['hour'] = result
        print(f"  預期筆數: {expected_count}")
        print(f"  觀測筆數: {observed_count}")
        print(f"  缺失筆數: {missing_count}")
        print(f"  缺失比例 (相對預期): {pct_expected:.2f}%")
        if pct_observed is not None:
            print(f"  缺失比例 (相對觀測): {pct_observed:.2f}%")
        else:
            print(f"  缺失比例 (相對觀測): N/A")
        
    def analyze_days(self):
        """分析天粒度的缺失"""
        print("\n[INFO] 分析天粒度缺失...")
        
        # 計算天數
        start_day = self.start_time.floor('D')
        end_day = self.end_time.floor('D')
        days_count = int((end_day - start_day).days) + 1
        
        # 建立完整天索引
        day_freq_index = pd.date_range(start=start_day, periods=days_count, freq=pd.Timedelta(days=1))
        
        expected_count = len(day_freq_index)
        
        # 對實際時間進行天對齁
        actual_days = self.df['LocalTime'].dt.floor('D').unique()
        observed_count = len(actual_days)
        
        missing_count = expected_count - observed_count
        pct_expected = (missing_count / expected_count * 100) if expected_count > 0 else 0
        pct_observed = (missing_count / observed_count * 100) if observed_count > 0 else None
        
        result = {
            'expected': expected_count,
            'observed': observed_count,
            'missing': missing_count,
            'pct_vs_expected': round(pct_expected, 2),
            'pct_vs_observed': round(pct_observed, 2) if pct_observed is not None else None,
            'duration_span': str(timedelta(days=expected_count - 1))
        }
        
        self.results['day'] = result
        print(f"  預期筆數: {expected_count}")
        print(f"  觀測筆數: {observed_count}")
        print(f"  缺失筆數: {missing_count}")
        print(f"  缺失比例 (相對預期): {pct_expected:.2f}%")
        if pct_observed is not None:
            print(f"  缺失比例 (相對觀測): {pct_observed:.2f}%")
        else:
            print(f"  缺失比例 (相對觀測): N/A")
        
    def print_summary(self):
        """輸出終端摘要"""
        print("\n" + "="*70)
        print("TIME COMPLETENESS ANALYSIS SUMMARY")
        print("="*70)
        print(f"檔案: {self.csv_path.name}")
        print(f"時間範圍: {self.start_time} 至 {self.end_time}")
        print(f"原始筆數: {self.data_points}")
        print(f"有效筆數: {self.data_points_cleaned}")
        print(f"移除筆數: {self.data_points - self.data_points_cleaned}")
        print("\n" + "-"*70)
        
        # 分鐘
        m = self.results['minute']
        print("【分鐘粒度】")
        print(f"  預期筆數: {m['expected']:,}")
        print(f"  觀測筆數: {m['observed']:,}")
        print(f"  缺失筆數: {m['missing']:,}")
        print(f"  缺失比例 (vs 預期): {m['pct_vs_expected']:.2f}%")
        if m['pct_vs_observed'] is not None:
            print(f"  缺失比例 (vs 觀測): {m['pct_vs_observed']:.2f}%")
        else:
            print(f"  缺失比例 (vs 觀測): N/A")
        print(f"  時間跨距: {m['duration_span']}")
        
        print("\n" + "-"*70)
        
        # 小時
        h = self.results['hour']
        print("【小時粒度】")
        print(f"  預期筆數: {h['expected']:,}")
        print(f"  觀測筆數: {h['observed']:,}")
        print(f"  缺失筆數: {h['missing']:,}")
        print(f"  缺失比例 (vs 預期): {h['pct_vs_expected']:.2f}%")
        if h['pct_vs_observed'] is not None:
            print(f"  缺失比例 (vs 觀測): {h['pct_vs_observed']:.2f}%")
        else:
            print(f"  缺失比例 (vs 觀測): N/A")
        print(f"  時間跨距: {h['duration_span']}")
        
        print("\n" + "-"*70)
        
        # 天
        d = self.results['day']
        print("【天粒度】")
        print(f"  預期筆數: {d['expected']:,}")
        print(f"  觀測筆數: {d['observed']:,}")
        print(f"  缺失筆數: {d['missing']:,}")
        print(f"  缺失比例 (vs 預期): {d['pct_vs_expected']:.2f}%")
        if d['pct_vs_observed'] is not None:
            print(f"  缺失比例 (vs 觀測): {d['pct_vs_observed']:.2f}%")
        else:
            print(f"  缺失比例 (vs 觀測): N/A")
        print(f"  時間跨距: {d['duration_span']}")
        
        print("\n" + "="*70)
        
    def save_json_report(self):
        """輸出 JSON 報告"""
        # 確保輸出目錄存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成帶時間戳的輸出檔名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"time_completeness_{self.csv_path.stem}_{timestamp}.json"
        output_path = self.output_dir / filename
        
        report = {
            'metadata': {
                'analysis_file': str(self.csv_path),
                'analysis_datetime': datetime.now().isoformat(),
                'time_range_start': self.start_time.isoformat(),
                'time_range_end': self.end_time.isoformat(),
            },
            'data_cleaning': {
                'original_count': self.data_points,
                'valid_count': self.data_points_cleaned,
                'removed_count': self.data_points - self.data_points_cleaned,
            },
            'minute_analysis': self.results['minute'],
            'hour_analysis': self.results['hour'],
            'day_analysis': self.results['day'],
            'notes': {
                'calculation_rules': [
                    'pct_vs_expected: 缺失筆數 / 預期筆數 * 100',
                    'pct_vs_observed: 缺失筆數 / 觀測筆數 * 100',
                    '首尾不完整時段已納入預期筆數計算',
                    '時間戳對齐至該粒度後去重計算',
                ]
            }
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8-sig') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\n[INFO] JSON 報告已保存: {output_path}")
            return output_path
        except Exception as e:
            print(f"[ERROR] 寫入 JSON 失敗: {e}")
            raise
            
    def run(self):
        """執行完整分析流程"""
        try:
            self.load_and_clean_data()
            self.analyze_minutes()
            self.analyze_hours()
            self.analyze_days()
            self.print_summary()
            self.save_json_report()
            print("\n[SUCCESS] 分析完成")
            return 0
        except Exception as e:
            print(f"\n[ERROR] 分析失敗: {e}", file=sys.stderr)
            return 1


def main():
    parser = argparse.ArgumentParser(
        description='分析 SolarRecord CSV 的 localTime 時間完整性',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='input/SolarRecord_260310_1829-row.csv',
        help='輸入 CSV 檔案路徑 (預設: input/SolarRecord_260310_1829-row.csv)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='輸出目錄 (預設: output)'
    )
    
    args = parser.parse_args()
    
    # 如果輸入路徑是相對路徑，以當前腳本所在目錄為基準
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path(__file__).parent / input_path
    
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path
    
    analyzer = TimeCompletenessAnalyzer(str(input_path), str(output_path))
    exit_code = analyzer.run()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
