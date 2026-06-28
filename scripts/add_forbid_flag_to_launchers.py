#!/usr/bin/env python3
"""
Add --forbid_model_override True and --on_override_action fail to launcher base_cmd or cmd lists.

Usage (dry-run):
  python scripts/add_forbid_flag_to_launchers.py --folder Power_day_autoTs_Prophet_6vB1 --pattern "*Lancher*.py" --dry-run

Apply changes:
  python scripts/add_forbid_flag_to_launchers.py --folder Power_day_autoTs_Prophet_6vB1 --pattern "*Lancher*.py"
"""
import argparse
import os
import fnmatch
import shutil
import re
import difflib


def find_matching_bracket(s, start):
    assert s[start] == '['
    i = start
    depth = 0
    in_single = False
    in_double = False
    esc = False
    while i < len(s):
        ch = s[i]
        if esc:
            esc = False
        elif ch == '\\':
            esc = True
        elif in_single:
            if ch == "'" and not esc:
                in_single = False
        elif in_double:
            if ch == '"' and not esc:
                in_double = False
        else:
            if ch == "'":
                in_single = True
            elif ch == '"':
                in_double = True
            elif ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def process_file(path, backup=True, dry_run=True):
    with open(path, 'r', encoding='utf-8') as f:
        src = f.read()
    new_src = src
    changed = False
    # find occurrences of cmd = [ or base_cmd = [
    for m in re.finditer(r"\b(cmd|base_cmd)\s*=\s*\[", src):
        b = src.find('[', m.start())
        if b == -1:
            continue
        end = find_matching_bracket(src, b)
        if end == -1:
            print(f"WARNING: unmatched bracket in {path} at pos {b}")
            continue
        list_str = src[b:end+1]
        if '--forbid_model_override' in list_str:
            continue
        # build insertion
        # try to preserve indentation
        nl = src.rfind('\n', 0, end)
        indent = ''
        if nl != -1:
            line = src[nl+1:end]
            indent = re.match(r'(\s*)', line).group(1)
        insert = ("\n" + indent + "    '--forbid_model_override', 'True',\n" +
                  indent + "    '--on_override_action', 'fail',")
        new_list = list_str[:-1] + insert + "\n" + indent + "]"
        new_src = new_src.replace(list_str, new_list, 1)
        changed = True
    if not changed:
        return False, None
    diff = '\n'.join(difflib.unified_diff(src.splitlines(), new_src.splitlines(), fromfile=path, tofile=path + '.modified', lineterm=''))
    if not dry_run:
        if backup:
            shutil.copy2(path, path + '.orig')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_src)
    return True, diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default='.', help='Folder to scan')
    parser.add_argument('--pattern', default='*Lancher*.py', help='Glob pattern for files to modify')
    parser.add_argument('--dry-run', action='store_true', default=False)
    parser.add_argument('--backup', action='store_true', default=True)
    args = parser.parse_args()

    files = []
    for root, dirs, filenames in os.walk(args.folder):
        for fname in filenames:
            if fnmatch.fnmatch(fname, args.pattern) and fname.endswith('.py'):
                files.append(os.path.join(root, fname))
    if not files:
        print("No files found matching pattern", args.pattern)
        return
    modified = []
    for f in sorted(files):
        ok, diff = process_file(f, backup=args.backup, dry_run=args.dry_run)
        if ok:
            modified.append((f, diff))
    print(f"Scanned {len(files)} files, {len(modified)} files require modification.")
    for f, diff in modified:
        print("==== File:", f)
        print(diff)
        print()
    if modified and not args.dry_run:
        print("Modifications applied. Original backups saved with `.orig` suffix.")

if __name__ == '__main__':
    main()
