# 10vA1 AutoTS 365-Day Forecast Launcher

此專案現已支援通過啟動器指令碼控制種子與輸出路徑。

## 檔案說明

### 主腳本
- **`10vA1_autots_365d_forecast.py`** - 核心 AutoTS 365 天預測程式
  - 新增 CLI 參數支援：
    - `--random_seed <seed>` - 設定隨機種子
    - `--output_dir <path>` - 覆蓋輸出根目錄
    - `--output_tag <tag>` - 在輸出目錄下新增標籤子目錄
    - `--train_csv <path>` - 覆蓋訓練資料 CSV 路徑
    - `--tmy_csv <path>` - 覆蓋 TMY 資料 CSV 路徑
  - 會另存一份 `effective_settings_<timestamp>.csv`，記錄本次使用的參數與 seed

### 啟動器
- **`10vA1_Lancher-S1_1Seed.py`** - 種子啟動器
  - 讀取種子 CSV 檔案（預設尋找 `autots_seeds_20260420_034832-1seed.csv`）
  - 支援種子索引選擇與自動續接
  - 統一管理輸出路徑與標籤
  - 逐筆種子迴圈執行主腳本

## 快速開始

### 1. 查看啟動器説明
```bash
python 10vA1_Lancher-S1_1Seed.py --help
```

### 2. 單筆執行（用第 1 個種子）
```bash
python 10vA1_Lancher-S1_1Seed.py --seeds_start_index 1
```

### 3. 指定種子檔案並從第 2 個種子開始
```bash
python 10vA1_Lancher-S1_1Seed.py --seeds_file input/autots_seeds_20260420_034832-30seed.csv --seeds_start_index 2
```

### 4. 自動續接（若已有執行過的種子，自動從下一個開始）
```bash
python 10vA1_Lancher-S1_1Seed.py --seeds_start_index auto
```

### 5. 使用 30-seed 檔案，自動續接
```bash
python 10vA1_Lancher-S1_1Seed.py --seeds_file input/autots_seeds_20260420_034832-30seed.csv --seeds_start_index auto
```

### 6. 從第 5 個種子開始執行
```bash
python 10vA1_Lancher-S1_1Seed.py --seeds_file input/autots_seeds_20260420_034832-30seed.csv --seeds_start_index 5
```

## Auto Resume 的詳細說明

**Auto Resume 工作原理**：
1. Launcher 在輸出目錄樹中搜尋最新的 `effective_settings.json` 檔案
2. 讀取其中的 `random_seed` 字段（該欄位記錄了上次執行的種子）
3. 在當前種子列表中找到該種子，計算其下一個種子的索引
4. 若下一個種子存在，從其開始執行；若已是最後一個種子，launcher 退出

**適用場景**：
- 長時間執行被中斷（如電源故障、手動停止）
- 想要批量執行多個種子，分次進行（例如分 3 天執行 30 個種子）

**注意事項**：
- 確保使用相同的種子檔案與輸出目錄，否則續接邏輯可能失效
- 若修改了種子檔案內容，建議從第 1 個開始重新執行，避免混淆

## 輸出目錄結構

預設輸出路徑為：
```
Power_1Year_Sum_10va1/
├── output/
│   └── default/
│       └── S1_1Seed/
│           ├── 10vA1_autots_365d_forecast_<timestamp>/
│           │   ├── forecast_365d_<timestamp>.csv
│           │   ├── model_metrics_<timestamp>.csv
│           │   ├── effective_settings_<timestamp>.csv
│           │   ├── training_log_<timestamp>.txt
│           │   ├── autots_model_<timestamp>.pkl
│           │   └── *.png (plots)
│           └── ...
```

## 種子檔案格式

支援兩種格式：
1. CSV 含 `SEED` 欄位：
   ```csv
   SEED
   1071865656
   1234567890
   ```

2. CSV 第一欄為種子數值：
   ```csv
   1071865656
   1234567890
   ```

## CLI 參數詳解

### `--seeds_file`
- 指定種子檔案路徑或名稱
- 支援相對路徑與絕對路徑
- 若無指定，依序搜尋：`AutoTs_SeedGen/autots_seeds*.csv` 或 `input/autots_seeds*.csv`
- 預設搜尋值由 `DEFAULT_SEEDS_FILE_NAME` 控制

### `--seeds_start_index`
- 指定種子起始位置，支援兩種模式：

#### Auto Resume 模式（`'auto'`）
- 自動尋找最新執行狀態中的 `effective_settings.json`
- 讀取其中的 `random_seed` 字段
- 找到該 seed 在列表中的位置，然後從**下一個** seed 開始執行
- 若已執行到最後一個 seed，launcher 會打印訊息並退出，不會重複執行
- 適合用於長時間執行被中斷後的**自動續接**

#### 數值模式
- `1` 或 `0` 表第 1 個種子
- `2` 表第 2 個種子
- `3` 表第 3 個種子
- 以此類推（1-based 語義轉換為 0-based 陣列索引）

**預設值**：`'auto'`

### `--output_dir`
- 指定輸出根目錄
- 預設：`output/default/<output_tag>`
- 支援相對與絕對路徑
- launcher 會建立此目錄（若不存在）

### `--output_tag`
- 輸出目錄下的標籤名稱
- 預設由啟動器檔名自動產生（去掉 `10vA1_Lancher-` 前綴）
- 例如 `10vA1_Lancher-S1_1Seed.py` → 標籤 `S1_1Seed`
- 若需自訂，可用 `--output_tag my_custom_tag` 指定

## 主腳本直接執行

主腳本也支援直接執行，無需啟動器：

```bash
python 10vA1_autots_365d_forecast.py --random_seed 12345 --output_dir ./my_output
```

此時會在 `./my_output/<timestamp>/` 下生成輸出檔案。
