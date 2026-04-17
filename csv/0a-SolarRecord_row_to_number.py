import argparse
import csv
import re
import os
from pathlib import Path
from datetime import datetime

try:
	from dateutil.parser import parse as date_parse
except Exception:
	date_parse = None


def parse_cell(s):
	if s is None:
		return ""
	s = s.strip()
	if s == "":
		return ""

	s_clean = s.replace(',', '')

	int_match = re.fullmatch(r'[+-]?\d+', s_clean)
	if int_match:
		try:
			return int(s_clean)
		except Exception:
			pass

	try:
		f = float(s_clean)
		return f
	except Exception:
		pass

	if date_parse:
		try:
			dt = date_parse(s, fuzzy=False)
			if 1800 <= dt.year <= 2100:
				if dt.time().hour == 0 and dt.time().minute == 0 and dt.time().second == 0 and dt.microsecond == 0:
					return dt.date().isoformat()
				return dt.isoformat(sep=' ')
		except Exception:
			pass

	return s


def convert_csv(input_path, output_path):
	input_p = Path(input_path)
	if not input_p.exists():
		alt = Path('csv') / input_p.name
		script_dir_alt = Path(__file__).parent / input_p.name
		if alt.exists():
			input_p = alt
		elif script_dir_alt.exists():
			input_p = script_dir_alt
		else:
			raise FileNotFoundError(f"Input file not found: {input_path}")

	with input_p.open('r', encoding='utf-8', newline='') as fin:
		reader = csv.reader(fin)
		rows = list(reader)

	out_rows = []
	for row in rows:
		out_row = [parse_cell(cell) for cell in row]
		out_rows.append(out_row)

	out_p = Path(output_path)
	if not out_p.parent.exists():
		try:
			out_p.parent.mkdir(parents=True, exist_ok=True)
		except Exception:
			pass

	with out_p.open('w', encoding='utf-8', newline='') as fout:
		writer = csv.writer(fout)
		for r in out_rows:
			writer.writerow(r)


def main():
	p = argparse.ArgumentParser(description='Convert CSV cells to numbers or dates')
	p.add_argument('input', nargs='?', default=os.path.join('csv', 'SolarRecord_260310_1829-row.csv'))
	p.add_argument('output', nargs='?', default=os.path.join('csv', 'SolarRecord_260310_1829-row-number.csv'))
	args = p.parse_args()
	convert_csv(args.input, args.output)


if __name__ == '__main__':
	main()

