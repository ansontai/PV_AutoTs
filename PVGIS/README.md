# PVGIS 轉檔說明

此資料夾包含從 PVGIS 原始時序（hourly）產生 daily 聚合，並將 daily 欄位對齊為專案所需 `SolarRecord` 格式的工具。

主要腳本
- `3v2-PVGIS_TimeseriesCsv_hourly_to_daily_And_MappingToMySolarRecord.py`：將 PVGIS hourly CSV 解析、轉換為每日聚合（輸出到 `PVGIS/output/`）。
- `map_pvgis_daily_to_solarrecord.py`：讀取 aggregator 產生的 daily CSV，將欄位重新命名/轉換為 `SolarRecord(260228)_d_forWh_WithCodis.csv` 相容欄位，輸出 `*-solarrecord-ready.csv`。
- `3v3-map_pvgis_daily_to_solarrecord.py`：啟動器，將指定的 raw 檔複製為 aggregator 預期檔名、執行 aggregator，接著執行 mapping 腳本（可使用 `--preview-map` 只預覽 mapping 結果）。

快速使用範例

1) 使用啟動器（從 raw 產生 daily 並 mapping）：

```bash
python PVGIS/3v3-map_pvgis_daily_to_solarrecord.py --raw PVGIS/raw/PVGIS/raw/tmy_24.148_120.703_2005_2023.csv
```

2) 若你已經有 daily CSV，直接預覽 mapping：

```bash
python PVGIS/map_pvgis_daily_to_solarrecord.py --input PVGIS/output/your-daily.csv --preview
```

3) 產出最終可用檔案：

```bash
python PVGIS/map_pvgis_daily_to_solarrecord.py --input PVGIS/output/your-daily.csv --output PVGIS/output/your-daily-solarrecord-ready.csv --force
```

注意事項
- 需要安裝 `pandas`、`numpy`。
- `3v3` 啟動器會將你的 raw 複製到 aggregator 之預期檔名；若該預期檔已存在，請使用 `--force` 覆寫或先備份原檔。
- mapping 腳本會檢查 `date` 與至少一個日能量欄位（`E_day_kWh` / `P_kWh`），若缺失會以錯誤終止。

如需我幫你用指定 raw 執行一次預覽（`--preview-map`），或調整欄位對照優先順序，告訴我要用哪個 raw 檔即可。 
