# PV_autoTs

簡短說明與專案重要設定紀錄。

## 工具與程式

### DataSumny Feature Analyzer

論文級 CSV 特徵分析與摘要生成工具。對 `DataSumny/input` 目錄內的所有 CSV 檔案進行完整的特徵分析，產出適合正式論文使用的統計摘要與報表。

**主程式**：`DataSumny/DataSumny_FeatureAnalyzer.py`

**使用方式**：
```bash
cd DataSumny
python DataSumny_FeatureAnalyzer.py --input-dir ./input --output-dir ./output --include-plots
```

**輸出包含**：
- 各檔案的完整分析（JSON 格式）
- 欄位詳細清單（CSV 格式，適合論文引用）
- 跨檔摘要表（CSV + Excel）
- 缺值與相關性圖表
- 分析報告

詳細文件：[DataSumny/README_FeatureAnalyzer.md](DataSumny/README_FeatureAnalyzer.md)

---

## `AUTOTS_AUTO_INSTALL` 環境變數說明

本專案包含一支輔助腳本 `run_autots_install_check.py`，可在檢查到缺少 `AutoTS` 時自動以 `pip` 在專案 `.venv` 中安裝所需套件。為了避免每次執行時不必要的自動安裝，該行為預設關閉，必須透過環境變數顯式啟用：

- 變數名稱：`AUTOTS_AUTO_INSTALL`（支援別名 `AUTOTS_INSTALL`）
- 啟用值（不區分大小寫）：`1`, `true`, `yes`, `y`

範例（PowerShell）：

```powershell
$env:AUTOTS_AUTO_INSTALL = '1'
& .venv\Scripts\python.exe run_autots_install_check.py
```

範例（bash / macOS / Linux）：

```bash
export AUTOTS_AUTO_INSTALL=1
source .venv/bin/activate
python run_autots_install_check.py
```

注意事項：
- 啟用後腳本會在偵測到缺少 `AutoTS` 時嘗試執行 `pip install`（包含升級 `pip setuptools wheel`、安裝 `AutoTS` 等），請在具備網路與必要權限的環境中使用。
- 若你在 CI、容器化環境或共享機器上執行，建議不要啟用此自動安裝機制，而是預先在目標環境中安裝好相依套件以確保可重現性。

如需改變預設行為或整合到啟動腳本，請檢視並修改 `run_autots_install_check.py`。
