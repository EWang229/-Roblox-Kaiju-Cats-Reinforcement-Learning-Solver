"""
Turn a plan into numbers.  These are the statistics behind the "insights" in the README: which cat carries the
score, who arrives last, how much money goes into Power Ups, how many turns of slack are left, and so on.
"""
from __future__ import annotations

from typing import Iterable, List

import pandas as pd

from . import rules as R
from .board import Board
from .plan import Plan
from .simulator import simulate


def describe(board: Board, plan: Plan, rules: R.Rules = R.Rules()) -> dict:
    """One flat dict of facts about how `plan` plays out on `board`."""
    res = simulate(board, plan, rules)
    big = max(range(3), key=lambda i: res.cat_scores[i])
    n = {k: 0 for k in range(7)}
    for _, cmd in plan.placements():
        n[cmd] += 1
    arrival = res.arrival_turn
    last_cat = res.arrival_order[-1] if res.arrival_order else None
    return {
        "board": board.name,
        "score": res.score,
        "cost": plan.cost,
        "big_cat": R.CAT_NAMES[big],
        "big_cat_share": round(res.cat_scores[big] / res.score, 3) if res.score else 0.0,
        "big_cat_arrives_last": last_cat == big,
        "big_cat_arrival_turn": arrival[big],
        "turns_of_slack": None if arrival[big] is None else rules.turns - arrival[big],
        "bed_order": "-".join(R.CAT_NAMES[i][0] for i in res.arrival_order),
        "power_ups": n[R.CMD_POWER_UP],
        "stomps": n[R.CMD_STOMP],
        "turn_commands": sum(n[k] for k in range(1, 5)),
        "money_on_power_ups": n[R.CMD_POWER_UP] * R.CMD_COST[R.CMD_POWER_UP],
        "bounce_floors": res.stats.get("bounce_floors", 0),
        "spike_hits": res.stats.get("spike_hits", 0),
        "mud_hits": res.stats.get("mud_hits", 0),
        "fights_lost": res.stats.get("fights_lost", 0),
        **{f"{R.CAT_NAMES[i].lower()}_score": s for i, s in enumerate(res.cat_scores)},
    }


def table(pairs: Iterable) -> pd.DataFrame:
    """DataFrame of :func:`describe` for an iterable of ``(board, plan)`` pairs."""
    return pd.DataFrame([describe(b, p) for b, p in pairs])


def markdown(df: pd.DataFrame, columns: List[str]) -> str:
    """Render selected columns as a GitHub markdown table (no extra dependencies such as tabulate)."""
    rows = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(f"{r[c]:,}" if isinstance(r[c], (int,)) and not isinstance(r[c], bool) else str(r[c])
                                      for c in columns) + " |")
    return "\n".join(rows)
