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

## L7_Lancher-S92b_G50_120d_TmyTrue_30Seed-FitEn
# T:\OneDrive\1TB\School\PV_autoTs\Power_1Year_Sum_10v2A\output\default\L7_Lancher-S92b_G50_120d_TmyTrue_30Seed-FitEn

## S99_G50_EsAll-30Seed
# Power_day_autoTs_Prophet_6vD1\output\default\S99_G50_EsAll-30Seed
default_input_path = str(Path("T:/OneDrive/1TB/School/PV_autoTs/Power_day_autoTs_Prophet_6vD1/output/default/S99_G50_EsAll-30Seed"))

## S9_G30_EsAll_90d-30Seed
# T:\OneDrive\1TB\School\PV_autoTs\Power_day_autoTs_Prophet_6vD1\output\default\S9_G30_EsAll_90d-30Seed
# default_input_path = str(Path("T:/OneDrive/1TB/School/PV_autoTs/Power_day_autoTs_Prophet_6vD1/output/default/S9_G30_EsAll_90d-30Seed"))

## S2_G30-30Seed
# T:\OneDrive\1TB\School\PV_autoTs\Power_day_autoTs_Prophet_6vD1\output\default\S2_G30-30Seed
# default_input_path = str(Path("T:/OneDrive/1TB/School/PV_autoTs/Power_day_autoTs_Prophet_6vD1/output/default/S2_G30-30Seed"))

## S0_G10-30Seed
# T:\OneDrive\1TB\School\PV_autoTs\Power_day_autoTs_Prophet_6vD1\output\default\S0_G10-30Seed
# default_input_path = str(Path("T:/OneDrive/1TB/School/PV_autoTs/Power_day_autoTs_Prophet_6vD1/output/default/S0_G10-30Seed"))

## S7_G10_EsAll-30Seed
# T:\OneDrive\1TB\School\PV_autoTs\Power_day_autoTs_Prophet_6vD1\output\default\S7_G10_EsAll-30Seed

## DatepartRegression_30seed
# T:\OneDrive\1TB\School\PV_autoTs\Power_day_autoTs_Prophet_6vB1\output\DatepartRegression\DatepartRegression_30seed
# default_input_path = str(Path("T:/OneDrive/1TB/School/PV_autoTs/Power_day_autoTs_Prophet_6vB1/output/DatepartRegression/DatepartRegression_30seed"))

## WR-14-30_30merge
# T:\OneDrive\1TB\School\PV_autoTs\Power_day_autoTs_Prophet_6vB1\output\WindowRegression\WR-14-30_30merge
# default_input_path = str(Path("T:/OneDrive/1TB/School/PV_autoTs/Power_day_autoTs_Prophet_6vB1/output/WindowRegression/WR-14-30_30merge"))

## MultivariateMotif-30Seed
# T:\OneDrive\1TB\School\PV_autoTs\Power_day_autoTs_Prophet_6vB1\6vB1R4_Lancher-MultivariateMotif-30Seed.py

## Theta_30seed
# T:\OneDrive\1TB\School\PV_autoTs\Power_day_autoTs_Prophet_6vB1\output\Theta\Theta_30seed

## UnobservedComponents-30Seed_260424_0333
# T:\OneDrive\1TB\School\PV_autoTs\Power_day_autoTs_Prophet_6vB1\output\UnobservedComponents-30Seed_260424_0333

## LastValueNaive_30seed-260424_022034
# T:\OneDrive\1TB\School\PV_autoTs\Power_day_autoTs_Prophet_6vB1\output\LastValueNaive\30seed-260424_022034
# default_input_path = str(Path("T:/OneDrive/1TB/School/PV_autoTs/Power_day_autoTs_Prophet_6vB1/output/LastValueNaive/30seed-260424_022034"))

## Prophet-30seed
# T:\OneDrive\1TB\School\PV_autoTs\Power_day_autoTs_Prophet_6vB1\output\Prophet-30seed-260420_092613
# default_input_path = str(Path("T:/OneDrive/1TB/School/PV_autoTs/Power_day_autoTs_Prophet_6vB1/output/Prophet-30seed-260420_092613"))


# t:\OneDrive\1TB\School\PV_autoTs\Power_day_autoTs_Prophet_6vB1\output\default\default_AllSettings\260425_145139\effective_settings.json
# default_input_path = str(Path("T:/OneDrive/1TB/School/PV_autoTs/Power_day_autoTs_Prophet_6vB1/output/default/default_AllSettings/"))

# default_input_path = str(Path("T:/OneDrive/1TB/School/PV_autoTs/Power_day_autoTs_Prophet_6vB1/output/ETS 30seed-260420_0752"))
# default_input_path = str(Path("T:/OneDrive/1TB/School/PV_autoTs/Power_day_autoTs_Prophet_6vB1/output/ETS 30seed-260420_0752"))

# default_output_path = str(Path(__file__).resolve().parent / 'output' / 'default_AllSettings' / 'default_AllSettings.csv')
# default_output_path = str(Path(__file__).resolve().parent / 'output' / 'DatepartRegression-30seed' / 'DatepartRegression-30seed.csv')
# default_output_path = str(Path(__file__).resolve().parent / 'output' / 'WindowRegression-WR-14-30_30merge' / 'WindowRegression-WR-14-30_30merge.csv')
sub_folder_name = Path(__file__).resolve().stem.replace('Launch_extract_metrics-', '')
default_output_path = str(Path(__file__).resolve().parent / 'output' / sub_folder_name / f'{sub_folder_name}.csv')

def main():
    # default_input_folder = (Path(__file__).resolve().parent.parent
    #                   / "Power_day_autoTs_Prophet_6vB1" / "output"
    #                   / "ETS 30seed-260420_0752")

    parser = argparse.ArgumentParser(description='Launch extract_forecast_metrics with sensible defaults')
    parser.add_argument('--folder', '-f', default=str(default_input_path), help='Folder path to search')
    parser.add_argument('--output', '-o', default=default_output_path, help='Output CSV path')
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
