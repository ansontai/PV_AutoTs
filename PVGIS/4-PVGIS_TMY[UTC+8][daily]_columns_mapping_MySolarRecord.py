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
    # 檢查 mapping_v2.csv
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(f"mapping 檔不存在：{MAPPING_FILE}")

    print(f"讀取映射規則：{MAPPING_FILE}")
    mapping = pd.read_csv(MAPPING_FILE)

    # 驗證必要欄位（mapping_v2 使用 'rule' 欄）
    required_cols = ['tmy_col', 'train_col']
    if not all(col in mapping.columns for col in required_cols):
        raise ValueError(f"mapping_v2.csv 必須包含欄位：{required_cols}")

    # 找並讀取輸入 CSV
    input_file = find_input_file()
    print(f"讀取輸入檔：{input_file}")
    df = pd.read_csv(input_file)

    print(f"輸入檔欄位數：{len(df.columns)}")
    print(f"映射規則數：{len(mapping)}")

    # env 用於安全評估 rule，逐步將原始欄位與衍生欄位放入 env
    env = {
        'to_datetime': pd.to_datetime,
        'sin': np.sin,
        'cos': np.cos,
        'pi': np.pi,
        'np': np,
        'pd': pd,
    }

    # 為避免欄位名含特殊字元造成 eval 失敗，建立原始欄位名 -> 安全變數名對照
    col_safe_map = {}
    for c in df.columns:
        safe = re.sub(r'[^0-9a-zA-Z_]', '_', c)
        if re.match(r'^[0-9]', safe):
            safe = 'c_' + safe
        col_safe_map[c] = safe
        env[safe] = df[c]

    # 預先提供常用衍生變數，例如 month（供 mapping rule 使用）
    if 'date' in df.columns:
        try:
            env['month'] = pd.to_datetime(df['date']).dt.month
        except Exception:
            env['month'] = None

    new_columns = {}

    # 按照 mapping 的順序逐條處理（允許前面產生的欄位被後面使用）
    for idx, row in mapping.iterrows():
        tmy_col = row.get('tmy_col')
        train_col = row.get('train_col')
        rule = row.get('rule') if 'rule' in row.index else None
        out_typ = row.get('output_typ') if 'output_typ' in row.index else None

        # 若有 rule，評估 expression；否則嘗試從原始欄位複製
        if pd.notna(rule) and str(rule).strip() != '':
            rule_text = str(rule).strip()
            # 支援兩種寫法：含等號則取等號右側；若沒有等號則直接當成表達式
            if '=' in rule_text:
                _, rhs = rule_text.split('=', 1)
            else:
                rhs = rule_text

            rhs = rhs.strip()
            # 使用安全欄位名替換原始欄位名（避免像 G(h)_kWhm2 這類無效變數名）
            # 先以長度排序避免部分字串被錯誤替換
            for orig in sorted(col_safe_map.keys(), key=len, reverse=True):
                safe = col_safe_map[orig]
                if orig in rhs:
                    rhs = rhs.replace(orig, safe)

            # 將 'x in {a,b}' 轉為向量化 'x.isin({a,b})'
            rhs = re.sub(r"\b(\w+)\s+in\s+(\{[^}]+\})", r"\1.isin(\2)", rhs)

            print(f"  評估 rule 為欄位 '{train_col}': {rhs}")

            try:
                # 禁用內建函式，僅提供 env
                result = eval(rhs, {'__builtins__': None}, env)
            except Exception as e:
                print(f"⚠ 評估 rule 失敗 ({train_col}): {e}")
                continue

            # 若是純量則重複為 Series
            if not hasattr(result, '__len__') or isinstance(result, (int, float, bool)):
                # 重複成與 df 長度一致的 Series
                result = pd.Series([result] * len(df))

            # 型別轉換
            if pd.notna(out_typ) and str(out_typ).strip() != '':
                typ = str(out_typ).strip().lower()
                try:
                    if typ == 'int':
                        result = result.astype('int')
                    elif typ == 'float':
                        result = result.astype('float')
                    elif typ == 'bool':
                        result = result.astype('bool')
                except Exception:
                    pass

            # 儲存並放入 env
            new_columns[train_col] = result
            env[train_col] = result

        else:
            # no rule -> copy from source column if available
            if tmy_col in df.columns:
                print(f"  複製欄位：{tmy_col} → {train_col}")
                series = df[tmy_col].copy()
                new_columns[train_col] = series
                env[train_col] = series
            else:
                print(f"  ⚠ 無 rule 且找不到來源欄位 {tmy_col}，跳過 {train_col}")

    # 將 new_columns 合併回原始 df（有相同名稱則覆寫）
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
