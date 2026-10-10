"""Chain-level benchmarking: how good is a search configuration, statistically?

A single search chain is a lottery ticket - on hard boards only a few percent of chains find the best basin.  So we judge
a configuration by the *distribution* of scores over many independent chains with a fixed time budget.  This is the
objective used by the Optuna tuner (:mod:`kaiju_cats.tune`).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import numpy as np
from joblib import Parallel, delayed

from . import rules as R
from .board import Board
from .solver import ALL_ORDERS, SolveConfig, _mix_seed, run_chain, warm_up


def chain_scores(board: Board, cfg: SolveConfig, n_chains: int, seconds: float, orders: Optional[Sequence] = None,
                 seed: int = 0, rules: R.Rules = R.Rules(), n_jobs: int = -1) -> List[int]:
    """Final (polished) score of `n_chains` independent fixed-length chains, cycling through `orders`."""
    warm_up(board, rules)
    orders = list(orders) if orders is not None else list(ALL_ORDERS)
    fixed = replace(cfg, min_seconds=seconds, plateau_seconds=1e9, max_seconds=seconds)
    jobs = [(orders[i % len(orders)], _mix_seed(seed, i)) for i in range(n_chains)]
    out = Parallel(n_jobs=n_jobs)(delayed(run_chain)(board, rules, o, s, fixed) for o, s in jobs)
    return [ch.score for ch in out]


def summarize(scores: Sequence[int], reference: Optional[int] = None) -> Dict[str, float]:
    a = np.asarray(scores, dtype=float)
    d = {"mean": a.mean(), "median": float(np.median(a)), "max": a.max(), "p90": float(np.percentile(a, 90))}
    if reference:
        d["hit_rate_99"] = float((a >= 0.99 * reference).mean())
    return d
