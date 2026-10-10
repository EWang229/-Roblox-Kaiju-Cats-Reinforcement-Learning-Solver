"""Solve the generated validation boards with the search solver (reference scores for the RL experiments)."""
import csv, sys, time
from dataclasses import replace
from pathlib import Path
from kaiju_cats import Board, SolveConfig, solve

root = Path(__file__).resolve().parents[1]
rows = []
for path in sorted((root / "boards_val").glob("*.txt")):
    board = Board.from_file(path)
    for seed in (12, 13):
        t = time.time()
        res = solve(board, replace(SolveConfig.preset("careful"), seed=seed))
        rows.append(dict(board=board.name, seed=seed, score=res.score, seconds=round(time.time() - t)))
        print(rows[-1], flush=True)
        res.plan.to_file(root / "boards_val" / f"{board.name}.seed{seed}.plan")
        with open(root / "results" / "val_search.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
