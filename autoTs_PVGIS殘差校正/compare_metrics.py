from pathlib import Path
import json
import sys


def find_latest_output(base_folder: Path):
    """尋找最新的 residual_correct_with_autots 輸出資料夾。

    這個函式會在 base_folder 下尋找
    autoTs_PVGIS殘差校正/output/ 目錄，並回傳最新修改時間的子資料夾。
    若找不到資料夾或 output 子目錄，會回傳 None。
    """
    out_root = base_folder / 'autoTs_PVGIS殘差校正' / 'output'
    if not out_root.exists():
        print(f'找不到 {out_root}，請確認已執行過 residual_correct_with_autots。')
        return None
    runs = [p for p in out_root.iterdir() if p.is_dir()]
    if not runs:
        print('找不到 output 子資料夾')
        return None
    latest = max(runs, key=lambda p: p.stat().st_mtime)
    return latest


def main():
    """主程式入口，讀取最新輸出資料夾並比較 PVGIS 與 Corrected 欄位分數。"""
    base = Path.cwd()
    latest = find_latest_output(base)
    if latest is None:
        sys.exit(1)

    print('使用最新 output:', latest)

    # 只讀取 PVGIS_vs_naiveLag1_metrics_*d.json 檔案
    files = sorted(latest.glob('PVGIS_vs_naiveLag1_metrics_*d.json'))
    if not files:
        print('在', latest, '找不到 metrics JSON 檔案')
        sys.exit(0)

    for f in files:
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
        except Exception as e:
            print('讀取', f, '錯誤：', e)
            continue

        pv = data.get('PVGIS', {})
        corr = data.get('Corrected', {}) or {}

        print('\n' + f.name)
        for metric_name, pv_value in pv.items():
            corrected_value = corr.get(metric_name)

            try:
                pv_float = float(pv_value) if pv_value is not None else None
                corr_float = float(corrected_value) if corrected_value is not None else None
            except Exception:
                pv_float = pv_value
                corr_float = corrected_value

            if pv_float is None or corr_float is None:
                print(f'  {metric_name}: PVGIS={pv_value}  Corrected={corrected_value}')
                continue

            # R2 為越大越好，其他誤差指標為越小越好
            if metric_name == 'R2':
                delta = corr_float - pv_float
                status = '↑較好' if delta > 0 else ('↓變差' if delta < 0 else '無變化')
                print(f'  {metric_name}: PVGIS={pv_float:.4f}  Corrected={corr_float:.4f}  ΔR2={delta:.4f}  ({status})')
            else:
                delta = corr_float - pv_float
                pct = ((pv_float - corr_float) / pv_float * 100) if pv_float != 0 else None
                status = '改善' if corr_float < pv_float else ('變差' if corr_float > pv_float else '無變化')
                pct_str = f'{pct:.1f}%' if pct is not None else 'N/A'
                print(f'  {metric_name}: PVGIS={pv_float:.4f}  Corrected={corr_float:.4f}  Δ={delta:.4f}  改善%={pct_str}  ({status})')


if __name__ == '__main__':
    main()
