"""No-network baselines on the real boards -> results/baselines.csv  (random play and per-board cross-entropy method)."""
import csv
from pathlib import Path

from kaiju_cats import Board, library
from kaiju_cats.rl.baselines import random_search, tabular_cem

root = Path(__file__).resolve().parents[1]
rows = []
for path in sorted((root / "boards").glob("*.txt")):
    b = Board.from_file(path)
    rows.append(dict(board=b.name, best_known=library.lookup(b)[1]["score"],
                     random_16k_games=random_search(b, 16384)[0], tabular_cem_20s=tabular_cem(b, seconds=20)[0]))
    print(rows[-1], flush=True)
with open(root / "results" / "baselines.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
