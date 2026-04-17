import requests
import pandas as pd
from io import StringIO

def fetch_pvgis_year(lat, lon, year, output_path=None):
    """
    從 PVGIS 取得一年資料，預設回傳 DataFrame。
    """
    url = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
    params = {
        "lat": lat,
        "lon": lon,
        "startyear": year,
        "endyear": year,
        "peakpower": 1,       # kW
        "loss": 14,           # system loss %
        "tracking": 0,        # 0: fixed, 1: one-axis, 2: two-axis
        "angle": 30,          # system tilt (度)
        "aspect": 180,        # system azimuth (度, 180 南向)
        "outputformat": "csv",
        "pvcalculation": 1,   # 伺服器計算PV發電
        "outputchoice": "json"  # 也可 csv
    }

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()

    # PVGIS 若 outputformat=csv，response.text是CSV內容
    csv_text = r.text
    
    # 可存檔
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(csv_text)

    # 轉成 DataFrame（若你要分析）
    df = pd.read_csv(StringIO(csv_text), comment='#')
    return df

if __name__ == "__main__":
    lat = 25.0330
    lon = 121.5654
    year = 2024
    out_file = "pvgis_2024_taipei.csv"

    df = fetch_pvgis_year(lat, lon, year, output_path=out_file)
    print("資料筆數:", len(df))
    print(df.head())
    print("已存檔:", out_file)