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
import re
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd


class TimeCompletenessAnalyzer:
    """時間完整性分析器"""
    
    def __init__(self, csv_path: str, output_dir: str = None):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir) if output_dir else self.csv_path.parent.parent / "output"
        self.df = None
        self.df_fixed_csv = None
        self.start_time = None
        self.end_time = None
        self.data_points = 0
        self.data_points_cleaned = 0
        self.results = {}

    def _normalize_localtime_text(self, value):
        """將 LocalTime 文字統一為 YYYY-MM-DD HH:MM:SS。"""
        if pd.isna(value):
            return ""

        raw = str(value).strip()
        if not raw:
            return ""

        if raw.lower() in {"nan", "nat", "none"}:
            return ""

        text = raw.replace("T", " ")

        # 斜線格式: YYYY/M/D H:M(:S)
        slash_match = re.match(
            r"^(\d{4})\/(\d{1,2})\/(\d{1,2})\s+(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?$",
            text,
        )
        if slash_match:
            year, month, day, hour, minute, second = slash_match.groups()
            second = second if second is not None else "00"
            return (
                f"{int(year):04d}-{int(month):02d}-{int(day):02d} "
                f"{int(hour):02d}:{int(minute):02d}:{int(second):02d}"
            )

        # 連字號格式: YYYY-M-D H:M(:S)
        hyphen_match = re.match(
            r"^(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?$",
            text,
        )
        if hyphen_match:
            year, month, day, hour, minute, second = hyphen_match.groups()
            second = second if second is not None else "00"
            return (
                f"{int(year):04d}-{int(month):02d}-{int(day):02d} "
                f"{int(hour):02d}:{int(minute):02d}:{int(second):02d}"
            )

        # 無法匹配的格式，保留原值讓 to_datetime 統一判定。
        return raw
        
    def _parse_localtime(self, ts_series):
        """
        解析 LocalTime 欄位。
        支援混合格式：
        - YYYY/M/D H:MM(:SS)
        - YYYY-MM-DD HH:MM(:SS)
        統一為 YYYY-MM-DD HH:MM:SS 後再解析。
        """
        normalized_text = ts_series.apply(self._normalize_localtime_text)
        parsed = pd.to_datetime(normalized_text, errors='coerce')
        return parsed, normalized_text
        
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
        
        # 過濾掉標題行混入資料的情況 (重置索引避免對齊問題)
        before_filter = len(self.df)
        self.df = self.df[self.df['LocalTime'].astype(str).str.strip() != 'LocalTime'].reset_index(drop=True)
        filtered_count = before_filter - len(self.df)
        if filtered_count > 0:
            print(f"[INFO] 過濾掉標題行混入: {filtered_count} 筆")
        
        # 保留原始字串，並先標準化後再解析
        self.df['LocalTime_raw'] = self.df['LocalTime'].astype(str).str.strip()
        parsed_localtime, normalized_text = self._parse_localtime(self.df['LocalTime_raw'])

        # 避免年份異常值（例如 2066）污染時間範圍統計。
        year_upper_bound = datetime.now().year + 1
        outlier_mask = parsed_localtime.notna() & (
            (parsed_localtime.dt.year < 2000) | (parsed_localtime.dt.year > year_upper_bound)
        )
        outlier_count = int(outlier_mask.sum())
        if outlier_count > 0:
            print(
                f"[WARNING] 偵測到年份異常筆數: {outlier_count} "
                f"(允許範圍 2000~{year_upper_bound})，將視為無效時間"
            )
            parsed_localtime.loc[outlier_mask] = pd.NaT

        self.df['LocalTime_norm_text'] = normalized_text
        self.df['LocalTime'] = parsed_localtime
        
        # 統計解析失敗的記錄
        invalid_mask = self.df['LocalTime'].isna()
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            print(f"[WARNING] 解析失敗筆數: {invalid_count} ({invalid_count/self.data_points*100:.2f}%)")
            # 顯示失敗樣本（原始值與標準化結果）
            failed_samples = self.df.loc[invalid_mask, ['LocalTime_raw', 'LocalTime_norm_text']].head(10)
            if len(failed_samples) > 0:
                print(f"[WARNING] 失敗樣本（前10筆）:")
                for idx, (_, row) in enumerate(failed_samples.iterrows(), 1):
                    print(f"         {idx}. raw='{row['LocalTime_raw']}' | normalized='{row['LocalTime_norm_text']}'")

        # 先建立統一格式 CSV 輸出資料（僅保留可解析列，維持原始順序）
        self.df_fixed_csv = self.df.loc[~invalid_mask].copy()
        self.df_fixed_csv['LocalTime'] = self.df_fixed_csv['LocalTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        self.df_fixed_csv = self.df_fixed_csv.drop(columns=['LocalTime_raw', 'LocalTime_norm_text'], errors='ignore')
        
        # 移除無效日期
        self.df = self.df.dropna(subset=['LocalTime']).reset_index(drop=True)
        self.data_points_cleaned = len(self.df)
        
        if self.data_points_cleaned == 0:
            raise ValueError("所有時間戳都無效或缺失（請檢查 LocalTime 欄位格式）")
        
        print(f"[INFO] 清理後筆數: {self.data_points_cleaned} (移除 {self.data_points - self.data_points_cleaned} 筆無效記錄)")
        
        # 排序並去重
        self.df = self.df.sort_values('LocalTime').drop_duplicates(subset=['LocalTime'], keep='last')
        self.data_points_cleaned = len(self.df)
        print(f"[INFO] 去重後筆數: {self.data_points_cleaned}")
        
        self.start_time = self.df['LocalTime'].min()
        self.end_time = self.df['LocalTime'].max()
        print(f"[INFO] 時間範圍: {self.start_time} 至 {self.end_time}")

    def save_fixed_date_csv(self):
        """輸出 LocalTime 已統一為 YYYY-MM-DD HH:MM:SS 的 CSV。"""
        if self.df_fixed_csv is None:
            raise RuntimeError("尚未建立 FixDate CSV 資料，請先執行 load_and_clean_data()")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self.csv_path.stem}[FixDate]{self.csv_path.suffix}"
        output_path = self.output_dir / filename

        try:
            self.df_fixed_csv.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"[INFO] FixDate CSV 已保存: {output_path}")
            return output_path
        except Exception as e:
            print(f"[ERROR] 寫入 FixDate CSV 失敗: {e}")
            raise
        
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
            self.save_fixed_date_csv()
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
