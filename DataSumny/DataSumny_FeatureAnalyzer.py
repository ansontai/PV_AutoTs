#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataSumny Feature Analyzer
論文級 CSV 特徵分析與摘要生成程式

對 DataSumny/input 內所有 CSV 檔案進行全面特徵提取與統計分析，
產出逐檔與跨檔總表，支援 CSV、JSON、Excel 與圖表輸出。

使用範例:
  python DataSumny_FeatureAnalyzer.py
  python DataSumny_FeatureAnalyzer.py --input-dir "./input" --output-dir "./output" --include-plots
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import argparse
import warnings

import pandas as pd
import numpy as np
from scipy import stats

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


class Logger:
    """簡單日誌系統"""
    def __init__(self, name: str = "DataSumny"):
        self.name = name
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                f'[{name}] [%(asctime)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)


class CSVMetadataExtractor:
    """單一 CSV 檔案的特徵提取器"""

    def __init__(self, df: pd.DataFrame, filename: str, logger: Logger):
        self.df = df
        self.filename = filename
        self.logger = logger

    def get_basic_statistics(self) -> Dict[str, Any]:
        """基礎統計特徵"""
        stats_dict = {
            "filename": self.filename,
            "shape_rows": int(self.df.shape[0]),
            "shape_columns": int(self.df.shape[1]),
            "memory_mb": float(self.df.memory_usage(deep=True).sum() / 1024 / 1024),
            "total_cells": int(self.df.shape[0] * self.df.shape[1]),
            "null_cells": int(self.df.isnull().sum().sum()),
            "null_ratio": float(self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "duplicate_ratio": float(self.df.duplicated().sum() / self.df.shape[0]) if self.df.shape[0] > 0 else 0.0,
        }
        return stats_dict

    def get_column_info(self) -> List[Dict[str, Any]]:
        """逐欄特徵分析"""
        column_infos = []

        for col in self.df.columns:
            series = self.df[col]
            col_type = str(series.dtype)

            # 基本欄位資訊
            col_info = {
                "column_name": str(col),
                "dtype": col_type,
                "non_null_count": int(series.notna().sum()),
                "null_count": int(series.isnull().sum()),
                "null_ratio": float(series.isnull().sum() / len(series)) if len(series) > 0 else 0.0,
                "unique_values": int(series.nunique(dropna=True)),
                "has_duplicates": bool(series.duplicated().any()),
                "is_constant": int(series.nunique(dropna=True)) <= 1,
            }

            # 檢測布林欄位
            is_bool = series.dtype in ['bool', 'boolean'] or (
                pd.api.types.is_numeric_dtype(series) and 
                len(series.dropna().unique()) <= 2 and 
                all(v in [0, 1, 0.0, 1.0, True, False] for v in series.dropna().unique())
            )

            # 數值欄特徵
            if pd.api.types.is_numeric_dtype(series) and not is_bool:
                try:
                    col_info.update({
                        "is_numeric": True,
                        "mean": float(series.mean()) if series.notna().any() else None,
                        "std": float(series.std()) if series.notna().any() else None,
                        "min": float(series.min()) if series.notna().any() else None,
                        "max": float(series.max()) if series.notna().any() else None,
                        "median": float(series.median()) if series.notna().any() else None,
                        "q25": float(series.quantile(0.25)) if series.notna().any() else None,
                        "q75": float(series.quantile(0.75)) if series.notna().any() else None,
                        "iqr": float(series.quantile(0.75) - series.quantile(0.25)) if series.notna().any() else None,
                        "skewness": float(stats.skew(series.dropna())) if len(series.dropna()) > 2 else None,
                        "kurtosis": float(stats.kurtosis(series.dropna())) if len(series.dropna()) > 2 else None,
                    })
                except Exception as e:
                    self.logger.debug(f"Failed to compute numeric stats for {col}: {e}")
                    col_info["is_numeric"] = True
            else:
                col_info["is_numeric"] = False
                # 字串欄特徵
                try:
                    value_counts = series.value_counts(dropna=True)
                    if len(value_counts) > 0:
                        col_info.update({
                            "most_common_value": str(value_counts.index[0]),
                            "most_common_freq": int(value_counts.iloc[0]),
                        })
                except Exception:
                    pass

            # 特殊類型檢測
            col_info.update(self._detect_special_types(series))

            column_infos.append(col_info)

        return column_infos

    def _detect_special_types(self, series: pd.Series) -> Dict[str, Any]:
        """偵測特殊欄位型態"""
        result = {
            "is_datetime": False,
            "is_binary": False,
            "is_boolean": False,
        }

        # 日期偵測
        if pd.api.types.is_datetime64_any_dtype(series):
            result["is_datetime"] = True
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sample = series.dropna().head(10)
                    if len(sample) > 0:
                        parsed = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)
                        if parsed.notna().sum() > 0:
                            result["is_datetime"] = True
            except Exception:
                pass

        # 二元/布林偵測
        try:
            if series.dtype in ['bool', 'boolean']:
                result["is_boolean"] = True
            elif pd.api.types.is_numeric_dtype(series):
                unique_vals = series.dropna().unique()
                if len(unique_vals) <= 2 and all(v in [0, 1, 0.0, 1.0] for v in unique_vals):
                    result["is_binary"] = True
        except Exception:
            pass

        return result

    def get_date_range(self) -> Optional[Dict[str, str]]:
        """若存在日期欄，回傳日期範圍"""
        for col in self.df.columns:
            if str(self.df[col].dtype).startswith('datetime'):
                try:
                    min_date = self.df[col].min()
                    max_date = self.df[col].max()
                    return {
                        "date_column": str(col),
                        "date_min": str(min_date),
                        "date_max": str(max_date),
                        "date_span_days": (max_date - min_date).days,
                    }
                except Exception:
                    pass

            # 嘗試將字串欄轉為日期
            try:
                parsed = pd.to_datetime(self.df[col].dropna(), errors='coerce')
                if parsed.notna().sum() > len(self.df) * 0.5:  # 至少 50% 可成功轉換
                    min_date = parsed.min()
                    max_date = parsed.max()
                    return {
                        "date_column": str(col),
                        "date_min": str(min_date),
                        "date_max": str(max_date),
                        "date_span_days": (max_date - min_date).days,
                    }
            except Exception:
                pass

        return None

    def get_timeseries_features(self) -> Optional[Dict[str, Any]]:
        """時間序列特徵（若有日期欄）"""
        date_info = self.get_date_range()
        if not date_info:
            return None

        ts_features = {"timeseries_info": date_info}

        # 找數值欄計算趨勢與波動
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            first_numeric = numeric_cols[0]
            series = self.df[first_numeric].dropna()
            if len(series) > 1:
                # 基本波動指標
                ts_features.update({
                    "numeric_col_used": str(first_numeric),
                    "variance": float(series.var()),
                    "cv": float(series.std() / series.mean()) if series.mean() != 0 else 0.0,
                    "trend_simple": float(np.polyfit(range(len(series)), series.values, 1)[0]),
                })

        return ts_features

    def get_correlation_summary(self) -> Optional[Dict[str, Any]]:
        """相關性與共線性分析摘要"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return None

        try:
            corr_matrix = numeric_df.corr(method='pearson')
            # 找高相關對（排除對角線）
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8:
                        high_corr_pairs.append({
                            "col1": str(corr_matrix.columns[i]),
                            "col2": str(corr_matrix.columns[j]),
                            "correlation": float(corr_val),
                        })

            # VIF 近似（針對第一個數值欄）
            vif_info = {}
            if numeric_df.shape[1] >= 2:
                try:
                    from numpy.linalg import matrix_rank
                    rank = matrix_rank(numeric_df.corr().values)
                    vif_info["effective_rank"] = int(rank)
                except Exception:
                    pass

            return {
                "numeric_columns": int(numeric_df.shape[1]),
                "high_correlation_pairs": high_corr_pairs[:5],  # 只保留前 5 對
                "correlation_matrix_mean_abs": float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()),
                **vif_info,
            }
        except Exception as e:
            self.logger.warning(f"Failed to compute correlation: {e}")
            return None

    def get_quality_metrics(self) -> Dict[str, Any]:
        """資料品質檢查指標"""
        metrics = {
            "completeness_ratio": float(self.df.notna().sum().sum() / (self.df.shape[0] * self.df.shape[1])),
            "constant_columns": int(sum(1 for col in self.df.columns if self.df[col].nunique(dropna=True) <= 1)),
            "problematic_columns": [],
        }

        # 標記問題欄位
        for col in self.df.columns:
            series = self.df[col]
            issues = []
            if series.isnull().sum() / len(series) > 0.5:
                issues.append("high_null_ratio")
            if series.nunique(dropna=True) <= 1:
                issues.append("constant")
            if series.duplicated().sum() / len(series) > 0.5:
                issues.append("high_duplicate_ratio")

            if issues:
                metrics["problematic_columns"].append({
                    "column": str(col),
                    "issues": issues,
                })

        # 極端偏態檢查 (只對非布林數值欄)
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 0:
            try:
                skewness_values = []
                for col in numeric_df.columns:
                    series = numeric_df[col].dropna()
                    # 排除布林類型
                    if len(series.unique()) > 2:
                        skew_val = stats.skew(series) if len(series) > 2 else 0
                        skewness_values.append(skew_val)
                
                if skewness_values:
                    extreme_skew = sum(1 for s in skewness_values if abs(s) > 2)
                    metrics["extreme_skew_columns"] = int(extreme_skew)
                else:
                    metrics["extreme_skew_columns"] = 0
            except Exception as e:
                self.logger.debug(f"Failed to compute skewness: {e}")
                metrics["extreme_skew_columns"] = 0

        return metrics

    def generate_complete_analysis(self) -> Dict[str, Any]:
        """整合所有特徵與分析"""
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "basic_statistics": self.get_basic_statistics(),
            "columns": self.get_column_info(),
            "date_range": self.get_date_range(),
            "timeseries_features": self.get_timeseries_features(),
            "correlation_summary": self.get_correlation_summary(),
            "quality_metrics": self.get_quality_metrics(),
        }
        return analysis


class DataSumnyAnalyzer:
    """主要分析引擎：協調多檔案分析與輸出"""

    def __init__(self, input_dir: Path, output_dir: Path, logger: Logger, include_plots: bool = False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.include_plots = include_plots

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def scan_input_files(self) -> List[Path]:
        """掃描輸入目錄內所有 CSV 檔案"""
        csv_files = sorted(self.input_dir.glob("*.csv"))
        self.logger.info(f"找到 {len(csv_files)} 個 CSV 檔案")
        for f in csv_files:
            self.logger.info(f"  - {f.name}")
        return csv_files

    def analyze_single_file(self, csv_path: Path) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
        """分析單一 CSV 檔案"""
        try:
            self.logger.info(f"正在分析: {csv_path.name}")
            df = pd.read_csv(csv_path, low_memory=False)
            extractor = CSVMetadataExtractor(df, csv_path.name, self.logger)
            analysis = extractor.generate_complete_analysis()
            return analysis, df
        except Exception as e:
            self.logger.error(f"讀取 {csv_path.name} 失敗: {e}")
            return {}, None

    def build_summary_table(self, all_analyses: List[Dict[str, Any]]) -> pd.DataFrame:
        """從所有分析結果構建總表"""
        summary_rows = []

        for analysis in all_analyses:
            if not analysis or "basic_statistics" not in analysis:
                continue

            basic = analysis["basic_statistics"]
            quality = analysis.get("quality_metrics", {})
            corr = analysis.get("correlation_summary", {})
            date_range = analysis.get("date_range", {})

            row = {
                "filename": basic.get("filename", ""),
                "rows": basic.get("shape_rows", 0),
                "columns": basic.get("shape_columns", 0),
                "memory_mb": basic.get("memory_mb", 0),
                "null_ratio": basic.get("null_ratio", 0),
                "duplicate_ratio": basic.get("duplicate_ratio", 0),
                "completeness": quality.get("completeness_ratio", 0),
                "constant_columns": quality.get("constant_columns", 0),
                "numeric_columns": corr.get("numeric_columns", 0),
                "extreme_skew_cols": quality.get("extreme_skew_columns", 0),
                "date_span_days": date_range.get("date_span_days", ""),
            }

            summary_rows.append(row)

        return pd.DataFrame(summary_rows)

    def save_analysis_json(self, analysis: Dict[str, Any], filename_stem: str) -> Path:
        """儲存 JSON 格式分析結果"""
        output_path = self.output_dir / f"{filename_stem}_analysis_{self.timestamp}.json"
        try:
            # 將 NaN/Inf 轉為 None 以便 JSON 序列化
            analysis_clean = self._make_json_serializable(analysis)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_clean, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已儲存 JSON: {output_path.name}")
            return output_path
        except Exception as e:
            self.logger.error(f"儲存 JSON 失敗: {e}")
            return None

    def save_analysis_csv(self, analysis: Dict[str, Any], filename_stem: str) -> Path:
        """儲存 CSV 格式分析結果（欄位列表）"""
        output_path = self.output_dir / f"{filename_stem}_columns_{self.timestamp}.csv"
        try:
            columns = analysis.get("columns", [])
            if columns:
                df = pd.DataFrame(columns)
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                self.logger.info(f"已儲存欄位 CSV: {output_path.name}")
                return output_path
        except Exception as e:
            self.logger.error(f"儲存欄位 CSV 失敗: {e}")
        return None

    def save_summary_table(self, summary_df: pd.DataFrame) -> Tuple[Optional[Path], Optional[Path]]:
        """儲存總表為 CSV 與 Excel"""
        csv_path = self.output_dir / f"summary_table_{self.timestamp}.csv"
        xlsx_path = self.output_dir / f"summary_table_{self.timestamp}.xlsx"

        try:
            summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            self.logger.info(f"已儲存摘要 CSV: {csv_path.name}")
        except Exception as e:
            self.logger.error(f"儲存摘要 CSV 失敗: {e}")
            csv_path = None

        # 若有 openpyxl，也產 Excel
        try:
            with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            self.logger.info(f"已儲存摘要 Excel: {xlsx_path.name}")
        except Exception as e:
            self.logger.warning(f"無法產生 Excel（可能缺少 openpyxl）: {e}")
            xlsx_path = None

        return csv_path, xlsx_path

    def generate_plots(self, df: pd.DataFrame, filename_stem: str) -> List[Path]:
        """生成圖表"""
        plot_paths = []

        if not MATPLOTLIB_AVAILABLE or not self.include_plots:
            return plot_paths

        try:
            # 缺值矩陣圖
            fig, ax = plt.subplots(figsize=(12, 8))
            null_matrix = df.isnull().astype(int)
            if SEABORN_AVAILABLE:
                sns.heatmap(null_matrix.iloc[:min(100, len(null_matrix))], cbar=True, ax=ax)
            else:
                ax.imshow(null_matrix.iloc[:min(100, len(null_matrix))], cmap='RdYlGn_r', aspect='auto')
            ax.set_title(f"{filename_stem} - Missing Values Pattern")
            plot_path = self.output_dir / f"{filename_stem}_missing_pattern_{self.timestamp}.png"
            fig.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            plot_paths.append(plot_path)
            self.logger.info(f"已儲存缺值圖: {plot_path.name}")
        except Exception as e:
            self.logger.warning(f"生成缺值圖失敗: {e}")

        try:
            # 相關性熱圖
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] > 1:
                corr_matrix = numeric_df.corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                if SEABORN_AVAILABLE:
                    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, ax=ax)
                else:
                    im = ax.imshow(corr_matrix.values, cmap='coolwarm')
                    ax.set_xticks(range(len(corr_matrix.columns)))
                    ax.set_yticks(range(len(corr_matrix.columns)))
                    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
                    ax.set_yticklabels(corr_matrix.columns)
                    plt.colorbar(im, ax=ax)
                ax.set_title(f"{filename_stem} - Correlation Matrix")
                plot_path = self.output_dir / f"{filename_stem}_correlation_{self.timestamp}.png"
                fig.savefig(plot_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                plot_paths.append(plot_path)
                self.logger.info(f"已儲存相關性圖: {plot_path.name}")
        except Exception as e:
            self.logger.warning(f"生成相關性圖失敗: {e}")

        return plot_paths

    def run(self) -> None:
        """執行完整分析流程"""
        self.logger.info("=" * 60)
        self.logger.info("開始 DataSumny 特徵分析")
        self.logger.info("=" * 60)

        csv_files = self.scan_input_files()
        if not csv_files:
            self.logger.warning("未找到 CSV 檔案")
            return

        all_analyses = []
        all_dataframes = {}

        # 逐檔分析
        for csv_path in csv_files:
            analysis, df = self.analyze_single_file(csv_path)
            all_analyses.append(analysis)
            all_dataframes[csv_path.name] = df

            # 產出逐檔結果
            if analysis:
                filename_stem = csv_path.stem
                self.save_analysis_json(analysis, filename_stem)
                self.save_analysis_csv(analysis, filename_stem)

                if df is not None:
                    self.generate_plots(df, filename_stem)

        # 產出總表
        self.logger.info("\n正在生成跨檔摘要表...")
        summary_df = self.build_summary_table(all_analyses)
        csv_path, xlsx_path = self.save_summary_table(summary_df)

        # 產出完整分析報告
        self.logger.info("\n正在生成完整分析報告...")
        report = {
            "timestamp": datetime.now().isoformat(),
            "input_directory": str(self.input_dir),
            "output_directory": str(self.output_dir),
            "total_files_analyzed": len(csv_files),
            "successful_analyses": len([a for a in all_analyses if a]),
            "summary_table_rows": len(summary_df),
            "summary_table_columns": len(summary_df.columns),
        }
        report_path = self.output_dir / f"analysis_report_{self.timestamp}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self.logger.info(f"已儲存分析報告: {report_path.name}")

        self.logger.info("\n" + "=" * 60)
        self.logger.info("分析完成！")
        self.logger.info(f"輸出目錄: {self.output_dir}")
        self.logger.info("=" * 60)

    @staticmethod
    def _make_json_serializable(obj: Any) -> Any:
        """遞迴轉換物件為 JSON 可序列化形式"""
        if isinstance(obj, dict):
            return {k: DataSumnyAnalyzer._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [DataSumnyAnalyzer._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if not np.isnan(obj) and not np.isinf(obj) else None
        elif isinstance(obj, (float, int)):
            return obj if not (isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj))) else None
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif obj is None:
            return None
        else:
            return str(obj)


def main():
    """命令列主入點"""
    parser = argparse.ArgumentParser(
        description="DataSumny Feature Analyzer - 論文級 CSV 特徵分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python DataSumny_FeatureAnalyzer.py
  python DataSumny_FeatureAnalyzer.py --input-dir ./input --output-dir ./output
  python DataSumny_FeatureAnalyzer.py --include-plots
        """
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default='./input',
        help='輸入 CSV 目錄 (預設: ./input)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='輸出結果目錄 (預設: ./output)'
    )
    parser.add_argument(
        '--include-plots',
        action='store_true',
        help='是否產生圖表 (預設: False)'
    )

    args = parser.parse_args()

    logger = Logger("DataSumny")

    # 驗證輸入目錄
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"輸入目錄不存在: {input_path}")
        sys.exit(1)

    output_path = Path(args.output_dir)

    analyzer = DataSumnyAnalyzer(
        input_dir=input_path,
        output_dir=output_path,
        logger=logger,
        include_plots=args.include_plots
    )

    analyzer.run()


if __name__ == "__main__":
    main()
