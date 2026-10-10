"""
A proven ceiling for a board, from the exact single-cat searches.

Reasoning: whatever the full plan does, each cat's final power is at most what that cat could reach if the other two cats
did not exist (fights and stolen floors can only hurt it) and given the money it alone spends.  Combine the three
single-cat tables over every way of splitting the budget and every bed-arrival order and you get a number that NO plan can
beat.  It ignores that two cats cannot both break the same floor (e.g. both double on the same plant), so it is usually
generous - think of it as "the map's hard limit", not as a target.

    ceiling(board).value        -> the bound
    ceiling(board).per_cat      -> each cat's best solo route at full budget (its "potential")
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from . import rules as R
from .board import Board
from .exact_solo import best_solo_routes


@dataclass
class Ceiling:
    value: int                        # no plan can score more than this
    order: Tuple[int, ...]            # bed-arrival order that attains the bound
    split: Tuple[int, int, int]       # money per cat that attains the bound
    solo_power: List[int]             # per cat: best power on arrival, full budget, before the bonus
    solo_final: List[int]             # per cat: that power x5 (potential as the last cat)
    exact: bool                       # False if a search hit its node limit (bound would then not be proven)


def ceiling(board: Board, rules: R.Rules = R.Rules(), max_nodes: int = 400_000_000) -> Ceiling:
    steps = rules.budget // 10 + 1
    arrive: List[np.ndarray] = []
    stay: List[np.ndarray] = []
    exact = True
    for cat in range(3):
        a, _plan, st, _nodes, finished = best_solo_routes(board, cat, rules.budget, rules, max_nodes=max_nodes)
        arrive.append(a)
        stay.append(st)
        exact &= bool(finished)

    def contribution(cat: int, k: int, rank: int) -> int:
        """Best final power of `cat` spending at most $10k and entering its bed as arrival number `rank` (0, 1, 2)."""
        best = int(stay[cat][k])
        p = int(arrive[cat][k])
        if p >= 0:
            best = max(best, p + R.ARRIVAL_ADD_FIRST if rank == 0 else p * (R.ARRIVAL_MULT_SECOND if rank == 1
                                                                          else R.ARRIVAL_MULT_THIRD))
        return best

    best_val, best_order, best_split = -1, (0, 1, 2), (0, 0, 0)
    for order in itertools.permutations(range(3)):             # order[r] = the cat that arrives r-th
        rank_of = {cat: r for r, cat in enumerate(order)}
        table = [[contribution(cat, k, rank_of[cat]) for k in range(steps)] for cat in range(3)]
        for k0 in range(steps):
            for k1 in range(steps - k0):
                k2 = steps - 1 - k0 - k1
                v = table[0][k0] + table[1][k1] + table[2][k2]
                if v > best_val:
                    best_val, best_order, best_split = v, order, (10 * k0, 10 * k1, 10 * k2)
    full = steps - 1
    return Ceiling(int(best_val), best_order, best_split,
                   [int(arrive[c][full]) for c in range(3)],
                   [int(arrive[c][full]) * R.ARRIVAL_MULT_THIRD for c in range(3)], exact)
