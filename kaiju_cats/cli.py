"""Command line interface:  ``python -m kaiju_cats <command> ...``  (or ``kaiju-cats`` once installed)."""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import replace
from pathlib import Path

from . import library, rules as R
from .analysis import describe
from .board import Board
from .plan import Plan
from .simulator import simulate
from .solver import SolveConfig, solve, solve_for


def _board(arg: str) -> Board:
    return Board.from_file(arg)


def cmd_solve(a) -> int:
    board = _board(a.board)
    known = library.lookup(board)
    if known and not a.fresh:
        plan, entry = known
        print(f"Known board - best stored plan scores {entry['score']:,} (found by {entry['found_by']}).")
        if not a.improve:
            print(plan.grid())
            print(plan.checklist())
            return 0
    cfg = replace(SolveConfig.preset(a.preset), seed=a.seed, n_jobs=a.jobs)

    def announce(res):
        print(res.summary(), flush=True)

    res = solve_for(board, a.minutes, cfg, progress=True, on_improve=announce) if a.minutes else \
        solve(board, cfg, progress=True)
    print(res.summary())
    print(res.plan.grid())
    print(res.plan.checklist())
    if a.save:
        res.plan.to_file(a.save)
        print(f"plan written to {a.save}")
    if not a.no_record and library.record(board, res.plan, res.score, f"python:{a.preset}:seed{a.seed}"):
        print("(new best for this board - stored in the library)")
    if a.plot:
        from .render import plot_plan
        plot_plan(board, res.plan, a.plot)
        print(f"picture written to {a.plot}")
    return 0


def cmd_show(a) -> int:
    board = _board(a.board)
    plan = Plan.from_file(board, a.plan)
    res = simulate(board, plan, trace=a.trace)
    print(f"score {res.score:,}  cost ${res.cost}  cats {res.cat_scores}  bed order {res.arrival_order}")
    print(plan.grid())
    print(plan.checklist())
    for k, v in describe(board, plan).items():
        print(f"  {k:22s} {v}")
    if a.plot:
        from .render import plot_plan
        plot_plan(board, plan, a.plot)
    return 0


def cmd_library(a) -> int:
    entries = library.load()
    for key, e in sorted(entries.items(), key=lambda kv: kv[1]["name"]):
        print(f"{e['name']:12s} {e['score']:>9,}  ${e['cost']}  {key[:10]}  {e['found_by']}")
    return 0


def cmd_random(a) -> int:
    from .generate import random_board
    b = random_board(random.Random(a.seed), name=f"random{a.seed}")
    text = b.to_text()
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)
    return 0


def cmd_benchmark(a) -> int:
    from .benchmark import chain_scores, summarize
    board = _board(a.board)
    cfg = replace(SolveConfig.preset("careful"), n_jobs=a.jobs)
    scores = chain_scores(board, cfg, a.chains, a.seconds, seed=a.seed, n_jobs=a.jobs)
    print(sorted(scores, reverse=True))
    print(summarize(scores))
    return 0


def cmd_tune(a) -> int:
    from .tune import tune
    tune(_board(a.board), trials=a.trials, seconds=a.seconds, seed=a.seed)
    return 0


DEFAULT_POLICY = Path(__file__).resolve().parent / "data" / "policy.pt"


def cmd_rl_solve(a) -> int:
    from .rl.solve import rl_solve
    from .rl.train import load
    board = _board(a.board)
    net = load(a.policy)
    res = rl_solve(board, net, seconds=a.seconds, seed=a.seed, log=print if a.verbose else None)
    print(res.summary())
    print(res.plan.grid())
    print(res.plan.checklist())
    if a.save:
        res.plan.to_file(a.save)
    if not a.no_record and library.record(board, res.plan, res.score, f"rl:{a.seconds:.0f}s:seed{a.seed}"):
        print("(new best for this board - stored in the library)")
    return 0


def cmd_rl_train(a) -> int:
    import glob
    from .rl.exit import ExItConfig, train_exit
    val = [Board.from_file(f) for f in sorted(glob.glob(a.val_glob))] if a.val_glob else None
    cfg = ExItConfig(iterations=a.iterations, seed=a.seed, save_path=a.out)
    train_exit(cfg, val_boards=val, log=lambda s: print(s, flush=True))
    return 0


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="kaiju-cats", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("solve", help="find a high-scoring plan for a board file")
    s.add_argument("board")
    s.add_argument("--preset", choices=["fast", "careful", "thorough"], default="careful")
    s.add_argument("--minutes", type=float, default=0, help="keep re-solving with new seeds for this long")
    s.add_argument("--seed", type=int, default=12)
    s.add_argument("--jobs", type=int, default=-1)
    s.add_argument("--save", help="write the plan to this file")
    s.add_argument("--plot", help="write a picture (png) of the plan")
    s.add_argument("--fresh", action="store_true", help="ignore the stored library plan")
    s.add_argument("--improve", action="store_true", help="solve even when the library already has a plan")
    s.add_argument("--no-record", action="store_true", help="do not store the result in the library")
    s.set_defaults(fn=cmd_solve)

    s = sub.add_parser("show", help="simulate and describe a plan file")
    s.add_argument("board"), s.add_argument("plan")
    s.add_argument("--trace", action="store_true"), s.add_argument("--plot")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("library", help="list boards the library has solved")
    s.set_defaults(fn=cmd_library)

    s = sub.add_parser("random-board", help="generate a random board")
    s.add_argument("--seed", type=int, default=0), s.add_argument("--out")
    s.set_defaults(fn=cmd_random)

    s = sub.add_parser("benchmark", help="score distribution of independent search chains")
    s.add_argument("board")
    s.add_argument("--chains", type=int, default=12), s.add_argument("--seconds", type=float, default=10)
    s.add_argument("--seed", type=int, default=0), s.add_argument("--jobs", type=int, default=-1)
    s.set_defaults(fn=cmd_benchmark)

    s = sub.add_parser("tune", help="tune the search hyper-parameters with Optuna")
    s.add_argument("board")
    s.add_argument("--trials", type=int, default=20), s.add_argument("--seconds", type=float, default=10)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_tune)

    s = sub.add_parser("rl-solve", help="solve a board with the reinforcement-learning policy")
    s.add_argument("board")
    s.add_argument("--policy", default=str(DEFAULT_POLICY))
    s.add_argument("--seconds", type=float, default=120)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--save"), s.add_argument("--verbose", action="store_true")
    s.add_argument("--no-record", action="store_true")
    s.set_defaults(fn=cmd_rl_solve)

    s = sub.add_parser("rl-train", help="train the policy with expert iteration on generated boards")
    s.add_argument("--iterations", type=int, default=700), s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="runs/policy.pt")
    s.add_argument("--val-glob", default="boards_val/*.txt")
    s.set_defaults(fn=cmd_rl_train)

    a = p.parse_args(argv)
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
