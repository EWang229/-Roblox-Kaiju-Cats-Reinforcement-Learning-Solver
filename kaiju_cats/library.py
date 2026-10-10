"""
A tiny on-disk library of solved boards.

Boards repeat.  In practice the same layouts come back again and again, and re-solving a board you have already
solved is a waste of minutes you may not have.  The library maps a board's signature (a hash of its layout) to the
best plan found so far, so ``python -m kaiju_cats solve`` can answer instantly for any board it has seen.

Entries are verified on load: a plan is only returned if the simulator reproduces its stored score.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from . import rules as R
from .board import Board
from .plan import Plan
from .simulator import simulate

DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "library.json"


def load(path: Path = DEFAULT_PATH) -> Dict[str, dict]:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def save(entries: Dict[str, dict], path: Path = DEFAULT_PATH) -> None:
    Path(path).write_text(json.dumps(entries, indent=1, sort_keys=True) + "\n")


def lookup(board: Board, path: Path = DEFAULT_PATH, rules: R.Rules = R.Rules()) -> Optional[tuple]:
    """Return ``(Plan, entry)`` for a known board, or ``None``.  The stored score is re-checked by simulation."""
    entry = load(path).get(board.signature())
    if entry is None:
        return None
    plan = Plan.from_placements(board, {(r, c, f): cmd for r, c, f, cmd in entry["placements"]})
    if simulate(board, plan, rules).score != entry["score"]:
        return None            # stale entry (rules changed since it was stored)
    return plan, entry


def record(board: Board, plan: Plan, score: int, found_by: str, path: Path = DEFAULT_PATH) -> bool:
    """Store `plan` if it beats what the library already has for this board.  Returns True if stored."""
    entries = load(path)
    key = board.signature()
    if key in entries and entries[key]["score"] >= score:
        return False
    entries[key] = {
        "name": board.name,
        "score": int(score),
        "cost": plan.cost,
        "found_by": found_by,
        "board": board.to_text(),
        "placements": [[s.row, s.col, s.floor, cmd] for s, cmd in plan.placements()],
    }
    save(entries, path)
    return True
