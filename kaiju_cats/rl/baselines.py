"""
Baselines that use the same environment but NO neural network.

They answer "what does the network add?":

* ``random_search``  - best of N uniformly random (masked) games.  Shows how hard the board is for luck alone.
* ``tabular_cem``    - the cross-entropy method with one categorical distribution per *slot* (floor).
  It learns, for this one board, "put command X on floor Y", exactly like a plan, but stays inside the
  lazy-decision environment (so the budget and dominated commands are handled).  It cannot transfer to a
  new board: the table is specific to the slot layout.  The network can, which is its whole point.
"""
from __future__ import annotations

import time
from typing import Tuple

import numpy as np

from ..board import Board
from .batch import MAX_SLOTS, N_ACTIONS, BatchEnv


def _sample(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    c = probs.cumsum(axis=1)
    u = rng.random((probs.shape[0], 1)) * c[:, -1:]
    return (u > c).sum(axis=1).clip(0, N_ACTIONS - 1)


def random_search(board: Board, n_games: int = 4096, seed: int = 0) -> Tuple[int, np.ndarray]:
    env = BatchEnv([board], rollouts=n_games)
    rng = np.random.default_rng(seed)
    env.reset()
    while env.any_alive():
        _, _, mask = env.observe()
        a = _sample(mask.astype(np.float64) + 1e-12 * 0, rng)
        env.step(a)
    best = int(env.scores.argmax())
    return int(env.scores[best]), env.plans[best, :board.n_slots].copy()


def tabular_cem(board: Board, seconds: float = 30.0, games: int = 2048, elite_frac: float = 0.05,
                smooth: float = 0.5, seed: int = 0) -> Tuple[int, np.ndarray]:
    """Cross-entropy method with a per-slot categorical distribution.  Returns (best score, best plan)."""
    rng = np.random.default_rng(seed)
    table = np.full((MAX_SLOTS, N_ACTIONS), 1.0 / N_ACTIONS)
    env = BatchEnv([board], rollouts=games)
    best_score, best_plan = -1, np.zeros(board.n_slots, dtype=np.int8)
    t0 = time.time()
    while time.time() - t0 < seconds:
        env.reset()
        while env.any_alive():
            _, _, mask = env.observe()
            slots = env.pending_slots()
            probs = table[np.where(slots >= 0, slots, 0)] * mask
            probs = probs + 1e-9 * mask
            env.step(_sample(probs, rng))
        n_el = max(2, int(elite_frac * games))
        elite = np.argsort(-env.scores)[:n_el]
        if env.scores[elite[0]] > best_score:
            best_score = int(env.scores[elite[0]])
            best_plan = env.plans[elite[0], :board.n_slots].copy()
        # new table = frequency of each command per slot among elites (only slots that fired in a game count)
        counts = np.full((MAX_SLOTS, N_ACTIONS), 1e-3)
        fired = np.zeros(MAX_SLOTS)
        for g in elite:
            plan = env.plans[g]
            # a slot "fired" if any command was recorded or it was reached; plans store 0 for NONE, so use reached-ness
            for s in range(board.n_slots):
                counts[s, plan[s]] += 1.0
        counts /= counts.sum(axis=1, keepdims=True)
        table = (1 - smooth) * table + smooth * counts
    return best_score, best_plan
