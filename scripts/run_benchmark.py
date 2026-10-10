"""Solve every board in boards/ with several seeds and write results/benchmark.csv.

Usage: python scripts/run_benchmark.py [preset] [seeds...]
"""
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

from kaiju_cats import Board, SolveConfig, library, solve

preset = sys.argv[1] if len(sys.argv) > 1 else "careful"
seeds = [int(s) for s in sys.argv[2:]] or [12, 13, 14]
root = Path(__file__).resolve().parents[1]
out = root / "results"
out.mkdir(exist_ok=True)
rows = []
for path in sorted((root / "boards").glob("*.txt")):
    board = Board.from_file(path)
    known = library.lookup(board)
    best_known = known[1]["score"] if known else None
    for seed in seeds:
        t = time.time()
        res = solve(board, replace(SolveConfig.preset(preset), seed=seed))
        rows.append(dict(board=board.name, preset=preset, seed=seed, score=res.score, seconds=round(time.time() - t),
                         library_best=best_known))
        print(rows[-1], flush=True)
        with open(out / f"benchmark_{preset}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
