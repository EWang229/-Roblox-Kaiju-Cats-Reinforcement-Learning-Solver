"""Solve every board in a folder with the RL pipeline and write results/rl_<folder>_<seconds>s.csv.

Usage: python scripts/run_rl_benchmark.py <boards|boards_val> <seconds> [seeds...]
The policy was trained only on generated boards.  Note: the board-time settings (temperatures, expert sizes)
were tuned a little on a few of the real boards, so boards_val (never looked at) is the cleaner test.
"""
import csv
import sys
import time
from pathlib import Path

from kaiju_cats import Board, library
from kaiju_cats.rl.solve import rl_solve
from kaiju_cats.rl.train import load

folder, seconds = sys.argv[1], float(sys.argv[2])
seeds = [int(s) for s in sys.argv[3:]] or [0]
root = Path(__file__).resolve().parents[1]
net = load(root / "kaiju_cats" / "data" / "policy.pt")

reference = {}
val_csv = root / "results" / "val_search.csv"
if val_csv.exists():
    for r in csv.DictReader(open(val_csv)):
        reference[r["board"]] = max(reference.get(r["board"], 0), int(r["score"]))

rows = []
for path in sorted((root / folder).glob("*.txt")):
    board = Board.from_file(path)
    known = library.lookup(board)
    ref = known[1]["score"] if known else reference.get(board.name)
    for seed in seeds:
        t = time.time()
        res = rl_solve(board, net, seconds=seconds, seed=seed)
        rows.append(dict(board=board.name, seconds_budget=int(seconds), seed=seed, sample=res.sample_score,
                         adapted=res.adapt_score, final=res.score, reference=ref, took=round(time.time() - t)))
        print(rows[-1], flush=True)
        with open(root / "results" / f"rl_{folder}_{int(seconds)}s.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
