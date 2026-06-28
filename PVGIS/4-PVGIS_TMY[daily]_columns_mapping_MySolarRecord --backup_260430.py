"""
PVGIS TMY CSV 欄位重命名和單位轉換
輸入：tmy_24.148_120.703_2005_2023[UTC+8][daily].csv
輸出：根據 mapping.csv 進行欄位重命名和轉換
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

HERE = Path(__file__).parent
# MAPPING_FILE = HERE / "input" / "mapping.csv"
MAPPING_FILE = HERE / "input" / "mapping_v2.csv"

# 嘗試多個輸入位置
POSSIBLE_INPUT_PATHS = [
    # HERE / "my" / "tmy_24.148_120.703_2005_2023[UTC+8][daily].csv",
    HERE / "input" / "tmy_24.148_120.703_2005_2023[UTC+8][daily].csv",
    # HERE.parent / "autoTs_PVGIS季節性先驗" / "input" / "tmy_24.148_120.703_2005_2023[UTC+8][daily].csv",
    # HERE.parent / "autoTs_PVGIS殘差校正" / "input" / "tmy_24.148_120.703_2005_2023[UTC+8][daily].csv",
]

OUTPUT_DIR = HERE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def find_input_file():
    """尋找輸入 CSV 檔案"""
    for path in POSSIBLE_INPUT_PATHS:
        if path.exists():
            print(f"✓ 找到輸入檔：{path}")
            return path
    raise FileNotFoundError(f"找不到輸入檔，已搜索:\n" + "\n".join(str(p) for p in POSSIBLE_INPUT_PATHS))


def parse_value_convert(convert_str):
    """解析 value_convert 欄位，例如 '*3.6' 返回 ('multiply', 3.6)"""
    if not convert_str or pd.isna(convert_str):
        return None, None
    
    convert_str = str(convert_str).strip()
    if not convert_str:
        return None, None
    
    # 解析形式：*3.6, /2, +10 等
    match = re.match(r'^([\+\-\*/])(\d+\.?\d*)$', convert_str)
    if match:
        op = match.group(1)
        value = float(match.group(2))
        op_map = {'+': 'add', '-': 'subtract', '*': 'multiply', '/': 'divide'}
        return op_map[op], value
    
    return None, None


def apply_conversion(series, op, value):
    """對 Series 應用轉換"""
    if op is None:
        return series
    
    if op == 'multiply':
        return series * value
    elif op == 'divide':
        return series / value
    elif op == 'add':
        return series + value
    elif op == 'subtract':
        return series - value
    else:
        return series


def main():
    # 檢查 mapping.csv
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(f"mapping.csv 不存在：{MAPPING_FILE}")
    
    print(f"讀取映射規則：{MAPPING_FILE}")
    mapping = pd.read_csv(MAPPING_FILE)
    
    # 驗證必要欄位
    required_cols = ['tmy_col', 'train_col', 'value_convert']
    if not all(col in mapping.columns for col in required_cols):
        raise ValueError(f"mapping.csv 必須包含欄位：{required_cols}")
    
    # 找並讀取輸入 CSV
    input_file = find_input_file()
    print(f"讀取輸入檔：{input_file}")
    df = pd.read_csv(input_file)
    
    print(f"輸入檔欄位數：{len(df.columns)}")
    print(f"映射規則數：{len(mapping)}")
    
    # 建立新欄位 DataFrame
    new_columns = {}
    
    for idx, row in mapping.iterrows():
        tmy_col = row['tmy_col']
        train_col = row['train_col']
        value_convert = row['value_convert']
        
        # 驗證源欄位存在
        if tmy_col not in df.columns:
            print(f"⚠ 警告：源欄位 '{tmy_col}' 在輸入檔中不存在，跳過")
            continue
        
        # 取得源數據
        source_data = df[tmy_col].copy()
        
        # 應用轉換
        op, op_value = parse_value_convert(value_convert)
        if op:
            print(f"  轉換：{tmy_col} → {train_col} (操作：{op} {op_value})")
            source_data = apply_conversion(source_data, op, op_value)
        else:
            print(f"  映射：{tmy_col} → {train_col}")
        
        # 添加到新欄位
        new_columns[train_col] = source_data
    
    # 將新欄位添加到原始 DataFrame
    for col_name, col_data in new_columns.items():
        df[col_name] = col_data
    
    print(f"\n合併後欄位總數：{len(df.columns)}")
    
    # 生成輸出檔名
    input_stem = input_file.stem
    output_filename = f"{input_stem}[mapped].csv"
    output_path = OUTPUT_DIR / output_filename
    
    # 輸出 CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✓ 輸出檔已保存：{output_path}")
    print(f"  欄位數：{len(df.columns)}")
    print(f"  列數：{len(df)}")
    print(f"\n新增欄位：{list(new_columns.keys())}")


if __name__ == "__main__":
    main()
