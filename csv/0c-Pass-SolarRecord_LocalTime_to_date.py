import csv
from pathlib import Path

INPUT = Path('csv') / 'SolarRecord_260310_1829-row-number.csv'
OUTPUT = Path('csv') / 'SolarRecord_260310_1829-fixed_0c.csv'

def fix_header(input_p: Path, output_p: Path):
	with input_p.open('r', encoding='utf-8', newline='') as fin:
		reader = csv.reader(fin)
		rows = list(reader)

	if not rows:
		print('Empty input file')
		return

	header = rows[0]
	header = ['Date' if h == 'LocalTime' else h for h in header]

	with output_p.open('w', encoding='utf-8', newline='') as fout:
		writer = csv.writer(fout)
		writer.writerow(header)
		for r in rows[1:]:
			writer.writerow(r)

	print(f'Wrote: {output_p}')


if __name__ == '__main__':
	if not INPUT.exists():
		print(f'Input not found: {INPUT}')
	else:
		fix_header(INPUT, OUTPUT)

