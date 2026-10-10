"""Import plans found by the original C++ solver into the solution library (each is re-verified by simulation).

Usage: python scripts/import_cpp_plans.py <maps_dir> <plans_dir>
"""
import glob
import sys

from kaiju_cats import library
from kaiju_cats.board import Board
from kaiju_cats.fastsim import FastSim
from kaiju_cats.plan import Plan
from kaiju_cats.simulator import simulate

maps_dir, plans_dir = sys.argv[1], sys.argv[2]
boards = [Board.from_file(p) for p in sorted(glob.glob(f"{maps_dir}/*.txt"))]
for plan_path in sorted(glob.glob(f"{plans_dir}/*.plan")):
    for b in boards:
        try:
            plan = Plan.from_file(b, plan_path)
        except Exception:
            continue
        res = simulate(b, plan)
        if res.cost != plan.cost or plan.cost > 200:
            continue
        if res.score < 20000:
            continue
        if library.record(b, plan, res.score, f"cpp:{plan_path.split('/')[-1]}"):
            print(f"{b.name:10s} {res.score:8d}  <- {plan_path.split('/')[-1]}")
