import requests
import pandas as pd
import numpy as np
from autots import AutoTS

# ----------------------------
# 1) 下載 PVGIS TMY（JSON）
# PVGIS API 入口： https://re.jrc.ec.europa.eu/api/v5_3/tmy?...
# 工具名：tmy；版本化入口在官方 API 說明中列出。[2](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/getting-started-pvgis/api-non-interactive-service_en)
# ----------------------------
def fetch_pvgis_tmy_json(lat, lon, api_version="v5_3", startyear=None, endyear=None):
    base = f"https://re.jrc.ec.europa.eu/api/{api_version}/tmy"
    params = {
        "lat": lat,
        "lon": lon,
        "outputformat": "json",
    }
    if startyear is not None:
        params["startyear"] = int(startyear)
    if endyear is not None:
        params["endyear"] = int(endyear)

    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def _find_hourly_table_in_json(payload):
    """
    PVGIS TMY JSON 在不同版本/設定下 key 可能不同。
    這裡用 heuristic：找「list[dict] 且包含 time/date 與常見欄位」的表格。
    """
    candidates = []

    def walk(obj):
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            keys = {k.lower() for k in obj[0].keys()}
            has_time = any(("time" in k) or ("date" in k) for k in keys)
            has_known = any(k in keys for k in [
                "t2m", "rh", "g(h)", "gb(n)", "gd(h)", "ws10m", "wd10m", "sp"
            ])
            if has_time and has_known:
                candidates.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)

    walk(payload)
    if not candidates:
        raise ValueError(
            "找不到 PVGIS TMY 的 hourly table。請 print(payload.keys()) 檢查 JSON 結構。"
        )
    return candidates[0]

def tmy_json_to_df(payload):
    rows = _find_hourly_table_in_json(payload)
    df = pd.DataFrame(rows)

    # 找 time/date 欄位
    time_col = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None)
    if time_col is None:
        raise ValueError(f"找不到時間欄位，現有欄位：{list(df.columns)}")

    # PVGIS TMY 的 timestamp 通常是 UTC；官方也提醒 header 有 irradiance time offset。[1](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/pvgis-tools/pvgis-typical-meteorological-year-tmy-generator_en)
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()
    return df

# ----------------------------
# 2) 把 TMY 映射成你的資料索引上的「季節性先驗特徵」
# ----------------------------
def build_tmy_prior_features(tmy_df, target_index, cols=("G(h)", "T2m", "RH"), local_tz="Asia/Taipei"):
    """
    以「月-日-時」對應（MM-DD-HH），把 TMY (8760h) 投影到你的時間軸上。
    - target_index 若是 naive，預設用 local_tz 當地時間再轉 UTC 對齊 TMY
    - 若你的資料本來就是 UTC 或 tz-aware，會直接轉 UTC
    """
    # 先把 TMY 做成 key = MM-DD-HH
    tmy = tmy_df.copy()
    # 確保需要的欄位存在
    missing = [c for c in cols if c not in tmy.columns]
    if missing:
        raise ValueError(f"TMY 缺少欄位 {missing}。可用欄位：{list(tmy.columns)}")

    tmy_key = tmy.index.strftime("%m-%d-%H")
    tmy_keyed = tmy.loc[:, list(cols)].copy()
    tmy_keyed.index = tmy_key  # index 變成 key

    # 處理目標索引時區 → UTC
    idx = pd.DatetimeIndex(target_index)
    if idx.tz is None:
        idx_utc = idx.tz_localize(local_tz).tz_convert("UTC")
    else:
        idx_utc = idx.tz_convert("UTC")

    keys = idx_utc.strftime("%m-%d-%H")
    prior = tmy_keyed.reindex(keys)

    # 遇到 2/29（閏日）可能會 NaN，用 2/28 同小時替代（或你也可改成 3/1 平均）
    if prior.isna().any().any():
        keys_fix = np.where(
            (idx_utc.month == 2) & (idx_utc.day == 29),
            idx_utc.strftime("02-28-%H"),
            keys
        )
        prior = tmy_keyed.reindex(keys_fix)

    prior.index = idx  # 回到原本索引（保留你原本的 tz/naive 形式）
    # 重新命名欄位成 tmy_*
    prior.columns = [
        "tmy_" + c.replace("(h)", "").replace("(n)", "").replace("(", "").replace(")", "").replace(" ", "").replace("/", "_")
        for c in prior.columns
    ]
    return prior

# ----------------------------
# 3) 丟給 AutoTS（方式 A：把 prior 當成低權重的輔助序列）
# 官方教學提到：推薦把外部變數作為時間序列一起丟進去，並用 weights 降低其重要性。[3](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html)
# ----------------------------
def fit_autots_with_tmy_prior(df_target, lat, lon, forecast_length, frequency="H"):
    """
    df_target: pd.DataFrame，index=DatetimeIndex，至少包含一欄 'y' (你的目標)
    """
    payload = fetch_pvgis_tmy_json(lat, lon, api_version="v5_3")
    tmy_df = tmy_json_to_df(payload)

    prior = build_tmy_prior_features(
        tmy_df,
        df_target.index,
        cols=("G(h)", "T2m", "RH"),   # 你也可以換成 Gb(n), Gd(h), WS10m... [1](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/pvgis-tools/pvgis-typical-meteorological-year-tmy-generator_en)
        local_tz="Asia/Taipei",
    )

    # 組成多序列資料：y + prior
    df_wide = df_target.copy()
    df_wide = df_wide.join(prior, how="left")

    # 權重：目標 y 權重高，prior 權重低
    weights = {"y": 20}
    weights.update({c: 1 for c in prior.columns})

    model = AutoTS(
        forecast_length=forecast_length,
        frequency=frequency,
        ensemble="simple",
        max_generations=10,
        num_validations=2,
        validation_method="backwards",
        model_list="fast",   # 需要更準可用 "medium"/"all"
    )

    model = model.fit(df_wide, weights=weights)
    pred = model.predict()
    # 只取 y 的預測
    y_forecast = pred.forecast[["y"]]
    return model, y_forecast, prior, tmy_df

# ----------------------------
# 用法示例
# ----------------------------
if __name__ == "__main__":
    # 假設你有每小時資料 df_target，index 是時間，欄位 y 是目標
    # df_target = pd.DataFrame({"y": ...}, index=...)
    # 這裡只示範結構
    rng = pd.date_range("2025-01-01", periods=24*60, freq="H", tz="Asia/Taipei")
    df_target = pd.DataFrame({"y": np.random.rand(len(rng))}, index=rng)

    model, yhat, prior_train, tmy_df = fit_autots_with_tmy_prior(
        df_target=df_target,
        lat=24.15,   # 改成你的座標
        lon=120.67,
        forecast_length=48,
        frequency="H"
    )
    print(yhat.head())