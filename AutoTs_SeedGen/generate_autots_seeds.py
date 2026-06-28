#!/usr/bin/env python3
"""產生 autoTS 使用的整數種子並存成 CSV（預設 30 個）。"""

from __future__ import annotations

import argparse
import csv
import datetime
import random
from pathlib import Path
from typing import List, Optional


def generate_seeds(count: int = 30, base_seed: Optional[int] = None) -> List[int]:
    if base_seed is None:
        rng = random.SystemRandom()
    else:
        rng = random.Random(base_seed)
    return [rng.randint(0, 2**31 - 1) for _ in range(count)]


def default_output_path() -> Path:
    name = f"autots_seeds_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
    return Path(__file__).resolve().parent / name


def save_csv(path: str | Path, seeds: List[int]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed"])
        for s in seeds:
            w.writerow([s])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate CSV of integer seeds for autoTS")
    p.add_argument("-n", "--count", type=int, default=30, help="number of seeds to generate")
    p.add_argument("-o", "--output", default=None, help="output CSV path (optional)")
    p.add_argument("-s", "--seed", type=int, default=None, help="base seed for reproducibility (optional)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    seeds = generate_seeds(args.count, args.seed)
    out_path = args.output if args.output else default_output_path()
    save_csv(out_path, seeds)
    print(f"Wrote {len(seeds)} seeds to {out_path}")


if __name__ == "__main__":
    main()
