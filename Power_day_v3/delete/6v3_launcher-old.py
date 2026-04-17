"""Launcher for 6v3-autoTs_WeatherToDayWh.py

這個檔案允許你在頂端編輯下列參數，然後啟動原始腳本：
- `default_max_generations`
- `horizons`
- `InfiniteLoop`
- `default_model_list`
- `default_ensemble`
- `default_n_jobs`
- `default_transformer_list`
- `default_num_validations`

如果某個參數在此檔案未被設定（仍為 None），會保留原始 `6v3-autoTs_WeatherToDayWh.py` 的預設值。

使用方法：編輯下方參數後，執行：
  python run_6v3_launcher.py

實作說明：此啟動器會讀入原始腳本內容，逐行用正規表達式尋找並取代指定的常數定義，產生暫存腳本後以當前 Python 解譯器執行。
"""
from __future__ import annotations
import os
import sys
import re
import json
from datetime import datetime

# --- 在此區編輯參數 (設為 None 表示不覆寫) ---
# 範例：default_max_generations = 5
default_max_generations = None
# 範例：horizons = [30, 14, 7]
# horizons = None
horizons = [9, 6, 3]
# 範例：InfiniteLoop = False
# InfiniteLoop = None
InfiniteLoop = False
# 範例：default_model_list = ['ARIMA', 'RandomForest']
# default_model_list = None
default_model_list = 'superFast' # 預設模型組合（AutoTS 內建的快速測試組合）
# 範例：default_ensemble = ['auto','simple']
default_ensemble = ['simple'] # 預設 ensemble 方法（AutoTS 內建的簡單平均）
# 範例：default_n_jobs = 2
# default_n_jobs = None
default_n_jobs = 4
# 範例：default_transformer_list = ['DifferencedTransformer','Scaler']
# default_transformer_list = None
default_transformer_list = [
  "DifferencedTransformer", # 避免被「抹平」成水平線
  "Scaler", # 避免被「抹平」成水平線
  ## LSTM 常見的前處理（不一定適合本資料集，請自行評估）
  'MinMaxScaler',       # LSTM 必備
  'Detrend',            # 去趨勢
  'DatepartRegression', # 加入時間特徵（小時、星期、季節
  ],
# 範例：default_num_validations = 3
# default_num_validations = None
default_num_validations = 3

# --- 不需修改以下程式 ---
HERE = os.path.dirname(__file__)
TARGET_NAME = '6v3-autoTs_WeatherToDayWh.py'
TARGET_PATH = os.path.normpath(os.path.join(HERE, TARGET_NAME))

if not os.path.exists(TARGET_PATH):
    print(f'找不到目標腳本: {TARGET_PATH}')
    sys.exit(2)

# mapping: variable name -> override value from above if not None
overrides = {
    'default_max_generations': default_max_generations,
    'horizons': horizons,
    'InfiniteLoop': InfiniteLoop,
    'default_model_list': default_model_list,
    'default_ensemble': default_ensemble,
    'default_n_jobs': default_n_jobs,
    'default_transformer_list': default_transformer_list,
    'default_num_validations': default_num_validations,
}

def _repr_py(v):
    """Return a Python source representation suitable for injecting into the script."""
    return json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else repr(v)

def make_modified_script(src_path: str, overrides: dict) -> str:
    """Read src_path, replace top-level assignment lines for keys present in overrides (non-None),
    write to a temporary file and return its path.
    """
    with open(src_path, 'r', encoding='utf-8') as f:
        txt = f.read()

    changed = 0
    for name, val in overrides.items():
        if val is None:
            continue
        replacement_body = f"{name} = {_repr_py(val)}"
        # Find the start of a top-level assignment for this name
        m = re.search(rf'^\s*{re.escape(name)}\s*=', txt, flags=re.MULTILINE)
        if m:
            start = m.start()
            # capture indentation to preserve leading spaces
            indent_match = re.match(r'\s*', txt[start:])
            indent = indent_match.group(0) if indent_match else ''
            repl = indent + replacement_body

            # locate the RHS start (first non-space after '=')
            rhs_pos = m.end()
            # skip whitespace
            while rhs_pos < len(txt) and txt[rhs_pos].isspace():
                rhs_pos += 1

            # if RHS starts with an opening bracket, find matching closing bracket
            if rhs_pos < len(txt) and txt[rhs_pos] in ('[', '{', '('):
                open_ch = txt[rhs_pos]
                close_ch = { '[':']', '{':'}', '(':')' }[open_ch]
                depth = 0
                j = rhs_pos
                while j < len(txt):
                    ch = txt[j]
                    if ch == open_ch:
                        depth += 1
                    elif ch == close_ch:
                        depth -= 1
                        if depth == 0:
                            j += 1
                            break
                    j += 1
                # j points after the closing bracket (or EOF)
                txt = txt[:start] + repl + txt[j:]
                changed += 1
            else:
                # single-line assignment: replace until end of line
                line_end = txt.find('\n', m.end())
                if line_end == -1:
                    line_end = len(txt)
                txt = txt[:start] + repl + txt[line_end:]
                changed += 1
        else:
            # fallback: insert near the top (after first import)
            insert_pos = 0
            m2 = re.search(r"(^\s*from\s+[\w\.]+\s+import|^\s*import\s+[\w\.]+)", txt, flags=re.MULTILINE)
            if m2:
                insert_pos = m2.start()
            header = replacement_body + "\n"
            txt = txt[:insert_pos] + header + txt[insert_pos:]
            changed += 1

    ts = datetime.now().strftime('%y%m%d_%H%M%S')
    tmp_name = f'temp_run_{ts}.py'
    tmp_path = os.path.join(HERE, tmp_name)
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(txt)
    return tmp_path, changed

def run_script(path: str) -> int:
    """Run script at path using the same python interpreter and stream output."""
    python = sys.executable or 'python'
    args = [python, path]
    try:
        rc = os.spawnv(os.P_WAIT, python, args)
    except Exception:
        # fallback to subprocess
        import subprocess
        p = subprocess.Popen(args)
        p.wait()
        rc = p.returncode
    return rc

def main():
    tmp_path, changed = make_modified_script(TARGET_PATH, overrides)
    if changed:
        print(f'已在暫存腳本中覆寫 {changed} 個變數，執行：{tmp_path}')
    else:
        print('沒有參數被覆寫，將執行原始腳本的複本。')
    try:
        rc = run_script(tmp_path)
        print(f'腳本結束，返回碼: {rc}')
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

if __name__ == '__main__':
    main()
