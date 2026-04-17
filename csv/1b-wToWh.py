"""
將 CSV 中欄位 `w` 改名為 `Wh`，預設會先備份原檔再覆寫。
修改檔案頂端的 `INPUT_FILE` / `OUTPUT_FILE` 變數以方便重複使用。
"""
import shutil
from pathlib import Path
import pandas as pd
import sys

# 可修改的輸入/輸出設定（放在檔案頂端）
# 預設指向 `csv/` 目錄下的檔案
INPUT_FILE = Path('csv/SolarRecord_260310_1829-hour.csv')
# 如果要覆寫原檔，把 OUTPUT_FILE 設為 INPUT_FILE；或指定其他檔名
OUTPUT_FILE = Path('csv/SolarRecord_260310_1829-hour-Wh.csv')
# 是否建立備份（會複製成 `原檔名.csv.bak`）
BACKUP = True
BACKUP_SUFFIX = '.bak'


def main():
    input_path = INPUT_FILE if INPUT_FILE.is_absolute() else Path.cwd() / INPUT_FILE
    output_path = OUTPUT_FILE if OUTPUT_FILE.is_absolute() else Path.cwd() / OUTPUT_FILE

    if not input_path.exists():
        print(f"找不到輸入檔案: {input_path}")
        sys.exit(1)

    if BACKUP:
        backup_name = input_path.stem + input_path.suffix + BACKUP_SUFFIX
        backup_path = input_path.with_name(backup_name)
        shutil.copy(input_path, backup_path)
        print(f"已建立備份: {backup_path}")

    df = pd.read_csv(input_path)

    # 找出名稱為 W 或 w（不區分大小寫）的欄位
    w_candidates = [c for c in df.columns if c.lower() == 'w']
    if w_candidates and 'Wh' not in df.columns:
        src = w_candidates[0]
        df = df.rename(columns={src: 'Wh'})
        df.to_csv(output_path, index=False)
        print(f"已將欄位 '{src}' 改名為 'Wh'，並寫出 {output_path}")
    elif 'Wh' in df.columns:
        print("檔案已包含欄位 'Wh'，未做變更。")
    else:
        print("找不到欄位 'w' 或 'W'，未做變更。")


if __name__ == '__main__':
    main()
