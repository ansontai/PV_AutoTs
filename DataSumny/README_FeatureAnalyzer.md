# DataSumny Feature Analyzer

論文級 CSV 特徵分析與摘要生成工具

## 功能概述

本程式對 `DataSumny/input` 目錄內的所有 CSV 檔案進行完整的特徵分析和統計，產出適合正式論文使用的各種摘要與報表。

### 輸出內容

#### 1. 逐檔分析結果

**格式**：JSON + CSV + 圖表

每個輸入 CSV 檔案會產生：
- `{filename}_analysis_{timestamp}.json` — 完整分析結果（JSON 格式）
  - 基礎統計（行數、列數、記憶體、缺值率等）
  - 逐欄特徵（數據型別、統計量、唯一值數、缺值狀況）
  - 時間序列特徵（若有日期欄）
  - 相關性摘要（高相關欄位對、有效秩等）
  - 品質檢查指標（完整度、常數欄、問題欄位等）

- `{filename}_columns_{timestamp}.csv` — 欄位詳細清單（易於論文引用）
  - 欄位名稱、型別、缺值率、唯一值、統計量等

- `{filename}_missing_pattern_{timestamp}.png` — 缺值矩陣圖
  - 視覺化顯示資料的缺值分佈

- `{filename}_correlation_{timestamp}.png` — 相關性熱圖
  - 數值欄位間的相關係數矩陣

#### 2. 跨檔摘要表

**格式**：CSV + Excel

- `summary_table_{timestamp}.csv` — 所有檔案的統計摘要
  - 適合直接嵌入論文表格
  - 包含：檔案名、列數、欄數、缺值率、完整度、數值欄數、極端偏態欄數、日期跨度等

- `summary_table_{timestamp}.xlsx` — Excel 版本（可進一步編輯）

#### 3. 分析報告

- `analysis_report_{timestamp}.json` — 執行總結
  - 分析時間、處理檔案數、成功率等中繼資訊

## 快速開始

### 基本執行

```bash
cd DataSumny
python DataSumny_FeatureAnalyzer.py
```

預設會掃描 `./input` 目錄的 CSV，輸出到 `./output`。

### 命令列選項

```bash
# 指定自訂輸入/輸出目錄
python DataSumny_FeatureAnalyzer.py --input-dir /path/to/input --output-dir /path/to/output

# 包含圖表生成（預設不產圖，加快執行速度）
python DataSumny_FeatureAnalyzer.py --include-plots

# 組合使用
python DataSumny_FeatureAnalyzer.py --input-dir ./input --output-dir ./output --include-plots
```

## 特徵說明

### 基礎統計
- **行數 (rows)**：資料集記錄數
- **欄數 (columns)**：特徵維度
- **缺值率 (null_ratio)**：$\frac{\text{空值格子數}}{\text{總格子數}}$
- **重複率 (duplicate_ratio)**：完全重複行的比例
- **完整度 (completeness)**：$1 - \text{缺值率}$

### 欄位特徵
- **dtype**：資料型別（numeric, object, datetime64 等）
- **is_numeric**：是否為數值欄
- **mean, std, min, max, median**：分布統計
- **q25, q75, iqr**：四分位數與四分位距
- **skewness, kurtosis**：高階矩（偏態與峰度）
- **unique_values**：唯一值數量

### 時間序列特徵（若檔案包含日期欄）
- **date_column**：日期欄位名稱
- **date_range**：日期最小-最大值
- **date_span_days**：時間跨度（日數）
- **trend_simple**：一次多項式趨勢係數
- **cv (coefficient of variation)**：變異係數

### 相關性與共線性
- **numeric_columns**：數值欄數
- **high_correlation_pairs**：相關係數 > 0.8 的欄位對
- **correlation_matrix_mean_abs**：相關矩陣平均絕對值
- **effective_rank**：相關矩陣的有效秩（共線性指標）

### 品質檢查指標
- **completeness_ratio**：資料集整體完整度
- **constant_columns**：完全常數欄的數量
- **problematic_columns**：含有高缺值率、常數或高重複率的欄位
- **extreme_skew_columns**：極端偏態（|skewness| > 2）的數值欄數

## 論文引用建議

### 資料特性表

使用 `summary_table_{timestamp}.csv` 中的結果直接製成表格：

| 資料源 | 記錄數 | 特徵數 | 完整度 | 數值特徵 | 日期跨度 |
|-------|--------|--------|--------|---------|---------|
| 2000--202602-d.csv | 9,404 | 48 | 95.7% | 24 | — |
| SolarRecord(...).csv | 313 | 57 | 98.2% | 31 | 312 天 |

### 欄位詳細表

使用各檔案的 `{filename}_columns_{timestamp}.csv` 產生欄位統計表。

### 品質評估

在方法論中使用 JSON 分析結果中的 `quality_metrics` 與 `date_range` 說明資料的清潔度與時間覆蓋。

## 輸出檔案位置

```
DataSumny/
├── input/                          # 輸入 CSV 檔案
├── output/                         # 分析結果
│   ├── {filename}_analysis_{ts}.json       # 各檔案完整分析
│   ├── {filename}_columns_{ts}.csv         # 各檔案欄位清單
│   ├── {filename}_missing_pattern_{ts}.png # 各檔案缺值圖
│   ├── {filename}_correlation_{ts}.png     # 各檔案相關性圖
│   ├── summary_table_{ts}.csv               # 跨檔摘要（CSV）
│   ├── summary_table_{ts}.xlsx              # 跨檔摘要（Excel）
│   └── analysis_report_{ts}.json            # 執行報告
└── DataSumny_FeatureAnalyzer.py     # 本程式
```

## 系統要求

- Python 3.8+
- pandas, numpy, scipy（必須）
- matplotlib, seaborn（可選，用於圖表生成）
- openpyxl（可選，用於 Excel 輸出）

## 執行範例

```bash
# 簡單執行
$ python DataSumny_FeatureAnalyzer.py

# 輸出樣例（部分）
[DataSumny] [2026-05-04 08:00:56] [INFO] 開始 DataSumny 特徵分析
[DataSumny] [2026-05-04 08:00:56] [INFO] 找到 7 個 CSV 檔案
[DataSumny] [2026-05-04 08:00:56] [INFO]   - 2000--202602-d.csv
[DataSumny] [2026-05-04 08:00:56] [INFO]   - SolarRecord(...).csv
...
[DataSumny] [2026-05-04 08:01:20] [INFO] 分析完成！
[DataSumny] [2026-05-04 08:01:20] [INFO] 輸出目錄: output
```

## 常見問題

**Q: 為什麼有些圖表沒有生成？**
A: 確認你有加上 `--include-plots` 旗標，且 matplotlib/seaborn 已安裝。

**Q: 缺值率如何計算？**
A: 計算所有空值格子數除以資料集總格子數（行數 × 欄數）。

**Q: 為什麼相關性矩陣有 NaN？**
A: 若某欄全為空值或只有常數，相關係數無法計算，會顯示為 NaN。

**Q: 時間跨度為 0 是什麼意思？**
A: 該檔案沒有可解析的日期欄，或所有日期值相同。

## 版本與更新

- **v1.0** (2026-05-04)：初始版本
  - 基礎統計、欄位分析、時間序列特徵
  - 相關性與共線性分析
  - 品質檢查與異常偵測
  - CSV/JSON/Excel 多格式輸出
  - 圖表生成（缺值、相關性）

## 聯絡與建議

如有改進建議或發現問題，請於本專案內回報。

---

**最後修改日期**：2026年5月4日
