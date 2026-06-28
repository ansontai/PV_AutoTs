#!/usr/bin/env python3
"""Launcher for extract_forecast_metrics.py with project-default folder.

預設資料夾為相對於本檔案上兩層的
../Power_day_autoTs_Prophet_6vB1/output/ETS 30seed-260420_0752/
可透過命令列參數覆寫。
"""
import argparse
import logging
from pathlib import Path
import extract_forecast_metrics


def main():
    default_input_folder = (Path(__file__).resolve().parent.parent
                      / "Power_day_autoTs_Prophet_6vB1" / "output"
                      / "ETS 30seed-260420_0752")

    parser = argparse.ArgumentParser(description='Launch extract_forecast_metrics with sensible defaults')
    parser.add_argument('--folder', '-f', default=str(default_input_folder), help='Folder path to search')
    parser.add_argument('--output', '-o', default=str(Path(__file__).resolve().parent / 'output' / 'forecast_metrics_extracted.csv'), help='Output CSV path')
    parser.add_argument('--pattern', '-p', default='forecast_Wh_metrics_*d.json', help='Glob pattern to search (default forecast_Wh_metrics_*d.json)')
    parser.add_argument('--no-recursive', action='store_true', help='Do not search subfolders')
    parser.add_argument('--absolute-paths', action='store_true', help='Store absolute file paths in CSV')
    parser.add_argument('--encoding', default='utf-8-sig', help='CSV encoding (default utf-8-sig)')
    args = parser.parse_args()

    argv = [args.folder, '--output', args.output]
    argv.extend(['--pattern', args.pattern])
    if args.no_recursive:
        argv.append('--no-recursive')
    if args.absolute_paths:
        argv.append('--absolute-paths')
    if args.encoding and args.encoding != 'utf-8-sig':
        argv.extend(['--encoding', args.encoding])

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    extract_forecast_metrics.main(argv)


if __name__ == '__main__':
    main()
