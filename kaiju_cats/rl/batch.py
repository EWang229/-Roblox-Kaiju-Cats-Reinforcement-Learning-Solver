"""
A batched, numba-compiled version of the lazy-decision MDP, built for reinforcement learning.

Why a new environment when ``kaiju_cats/env.py`` (Gymnasium) already exists?
-----------------------------------------------------------------------------
Speed.  The Gymnasium env wraps the readable Python simulator: one game step costs tens of microseconds of
Python, and a neural network needs *millions* of games.  Here hundreds of games advance together inside
one compiled function, and the network sees a whole batch of decisions per forward pass.

The decision problem (same as env.py)
-------------------------------------
Play the game forward.  Whenever a cat destroys a floor, the agent must pick that floor's command
(none / N / E / S / W / Stomp / Power Up).  The game then continues until the next floor is destroyed.
An episode ends after 15 turns; the only reward is the final score.  Because every game is deterministic,
the same sequence of commands always gives the same score.

What is in this file
--------------------
* ``BoardBatch``     - several boards packed into stacked arrays so numba can index them.
* ``BatchEnv``       - B games running in lock-step (B = boards x rollouts-per-board).
* ``_advance`` / ``_step_all`` / ``_observe`` - the compiled kernels (they re-use the exact game logic of
  ``kaiju_cats.fastsim`` so the RL environment cannot drift from the verified simulator).
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from numba import njit

from .. import rules as R
from ..board import Board
from ..fastsim import (NST, S_ARRIVED, S_COL, S_DIR, S_EVC, S_EVCAT, S_EVPTR, S_EVR, S_EVSLOT, S_NEV, S_POWER,
                       S_ROW, S_SPENT, S_STATUS, S_TURN, ST_ACTIVE, ST_BED, ST_GONE, ST_STOMP, ST_STUCK, _COST,
                       all_done, apply_event, init_state, move_phase, total_score)

MAX_SLOTS = 48            # most slots any board can have (24 buildings x 2 floors); used to pad plan arrays
MAX_DECISIONS = 64        # a game can never need more decisions than this (<= 3 cats x 15 turns, in practice ~15)
N_CHANNELS = 18           # image channels the network sees (see _observe)
N_VECTOR = 41             # scalar features the network sees (see _observe)
N_ACTIONS = R.N_COMMANDS


class BoardBatch:
    """Boards stacked into plain arrays: the form the compiled kernels (and a GPU, if you ever want one) like."""

    def __init__(self, boards: Sequence[Board]):
        self.boards: List[Board] = list(boards)
        n = len(self.boards)
        self.tiles = np.zeros((n, R.ROWS, R.COLS), dtype=np.int64)
        self.bed_owner = np.zeros((n, R.ROWS, R.COLS), dtype=np.int64)
        self.sr = np.zeros((n, 3), dtype=np.int64)
        self.sc = np.zeros((n, 3), dtype=np.int64)
        self.prio = np.zeros((n, 3), dtype=np.int64)
        self.slot_base = np.zeros((n, R.ROWS, R.COLS), dtype=np.int64)
        self.n_slots = np.zeros(n, dtype=np.int64)
        for i, b in enumerate(self.boards):
            tiles, bed_owner, sr, sc, prio, slot_base, _, _ = b.numba_pack()
            self.tiles[i], self.bed_owner[i], self.sr[i], self.sc[i] = tiles, bed_owner, sr, sc
            self.prio[i], self.slot_base[i], self.n_slots[i] = prio, slot_base, b.n_slots

    def __len__(self) -> int:
        return len(self.boards)


# =============================================================================================
# Compiled kernels
# =============================================================================================
@njit(cache=True)
def _pack(tiles, bed_owner, sr, sc, prio, slot_base, dummy, b):
    """Rebuild the 8-tuple that fastsim's functions expect, for board number `b` of the batch."""
    return (tiles[b], bed_owner[b], sr[b], sc[b], prio[b], slot_base[b], dummy, dummy)


@njit(cache=True)
def _advance(st, fl, pack, flags):
    """Play the game forward until it needs a decision.  True = decision pending, False = game over."""
    turns = flags[3]
    while True:
        if st[S_EVPTR] < st[S_NEV]:
            return True
        if all_done(st) or st[S_TURN] >= turns:
            return False
        move_phase(st, fl, pack, flags)


@njit(cache=True)
def _reset_all(ST, FL, bidx, tiles, bed_owner, sr, sc, prio, slot_base, dummy, flags, alive, plans):
    for g in range(ST.shape[0]):
        pack = _pack(tiles, bed_owner, sr, sc, prio, slot_base, dummy, bidx[g])
        init_state(ST[g], FL[g], pack, flags)
        alive[g] = _advance(ST[g], FL[g], pack, flags)
        for s in range(plans.shape[1]):
            plans[g, s] = 0


@njit(cache=True)
def _step_all(ST, FL, bidx, tiles, bed_owner, sr, sc, prio, slot_base, dummy, flags, actions, alive, plans, scores):
    """Apply `actions[g]` to the pending decision of every live game and advance to the next decision."""
    for g in range(ST.shape[0]):
        if not alive[g]:
            continue
        st, fl = ST[g], FL[g]
        pack = _pack(tiles, bed_owner, sr, sc, prio, slot_base, dummy, bidx[g])
        e = st[S_EVPTR]
        cmd = actions[g]
        if st[S_SPENT] + _COST[cmd] > flags[4]:       # the policy's mask should make this impossible; be safe
            cmd = 0
        plans[g, st[S_EVSLOT + e]] = cmd
        apply_event(st, fl, pack, flags, e, cmd)
        st[S_EVPTR] = e + 1
        alive[g] = _advance(st, fl, pack, flags)
        if not alive[g]:
            scores[g] = total_score(st)


@njit(cache=True)
def _log_power(p):
    return min(1.2, np.log(1.0 + p) / np.log(2.0e6))


@njit(cache=True)
def _observe(ST, FL, bidx, tiles, bed_owner, flags, alive, img, vec, mask):
    """Write what the network sees for every live game.

    Image (N_CHANNELS x 5 x 7), always from the point of view of the cat that just broke a floor ("self"):
        0-8   tile type one-hot (nothing, grass, gray bldg, yellow bldg, plant, mud, spikes, boulder, bed)
        9     floors still standing on the tile (/2)
        10    where *self* is          11, 12  where the two other cats are (lower cat id first)
        13    self's bed               14, 15  the other cats' beds
        16,17 row / column coordinates (so the net can tell edges and corners from the middle)
    Vector (N_VECTOR): turn, money left, which cat is self, the broken floor's type and what is left of it,
        then for self and each other cat: status (5), facing (4) and log-power (1), and bed arrivals so far.
    Mask (7): which of the commands are worth considering.
        * unaffordable commands are out;
        * a direction equal to the cat's current direction does nothing but cost money, so it is out;
        * Stomp is only useful if another floor is left on the tile.
    """
    nrow, ncol = R.ROWS, R.COLS
    for g in range(ST.shape[0]):
        if not alive[g]:
            continue
        st, fl, b = ST[g], FL[g], bidx[g]
        e = st[S_EVPTR]
        me = st[S_EVCAT + e]
        er, ec = st[S_EVR + e], st[S_EVC + e]
        o1 = 0
        o2 = 0
        k = 0
        for c in range(3):
            if c != me:
                if k == 0:
                    o1 = c
                else:
                    o2 = c
                k += 1
        for ch in range(N_CHANNELS):
            for r in range(nrow):
                for c in range(ncol):
                    img[g, ch, r, c] = 0.0
        for r in range(nrow):
            for c in range(ncol):
                t = tiles[b, r, c]
                img[g, t, r, c] = 1.0
                img[g, 9, r, c] = fl[r, c] / 2.0
                img[g, 16, r, c] = r / (nrow - 1.0)
                img[g, 17, r, c] = c / (ncol - 1.0)
                if t == R.T_BED:
                    owner = bed_owner[b, r, c]
                    if owner == me:
                        img[g, 13, r, c] = 1.0
                    elif owner == o1:
                        img[g, 14, r, c] = 1.0
                    else:
                        img[g, 15, r, c] = 1.0
        cats = (me, o1, o2)
        for j in range(3):
            c = cats[j]
            if st[S_STATUS + c] != ST_GONE:
                img[g, 10 + j, st[S_ROW + c], st[S_COL + c]] = 1.0
        # ---- vector ------------------------------------------------------------------------
        for i in range(N_VECTOR):
            vec[g, i] = 0.0
        vec[g, 0] = st[S_TURN] / flags[3]
        vec[g, 1] = (flags[4] - st[S_SPENT]) / flags[4]
        vec[g, 2 + me] = 1.0
        t = tiles[b, er, ec]
        if t == R.T_LOW:
            vec[g, 5] = 1.0
        elif t == R.T_HIGH:
            vec[g, 6] = 1.0
        else:
            vec[g, 7] = 1.0
        vec[g, 8] = fl[er, ec] / 2.0
        base = 9
        for j in range(3):
            c = cats[j]
            vec[g, base + st[S_STATUS + c]] = 1.0
            vec[g, base + 5 + st[S_DIR + c]] = 1.0
            vec[g, base + 9] = _log_power(st[S_POWER + c])
            base += 10
        # base == 39
        vec[g, 39] = st[S_ARRIVED] / 3.0
        vec[g, 40] = 1.0 if st[S_ARRIVED] >= 2 else 0.0
        # ---- action mask -----------------------------------------------------------------------
        left = flags[4] - st[S_SPENT]
        for a in range(N_ACTIONS):
            ok = _COST[a] <= left
            if a >= 1 and a <= 4 and st[S_DIR + me] == a - 1:
                ok = False
            if a == R.CMD_STOMP and fl[er, ec] <= 0:
                ok = False
            mask[g, a] = ok
        mask[g, 0] = True


# =============================================================================================
# Python wrapper
# =============================================================================================
class BatchEnv:
    """B games in lock-step.  ``boards`` x ``rollouts`` games: every board is played ``rollouts`` times.

    Typical loop::

        env = BatchEnv(boards, rollouts=32)
        env.reset()
        while env.any_alive():
            obs = env.observe()                 # img, vec, mask for the live games
            env.step(actions)                   # one action per game (ignored for finished games)
        env.scores, env.plans                   # final scores and the command plan of every game
    """

    def __init__(self, boards: Sequence[Board], rollouts: int = 1, rules: R.Rules = R.Rules()):
        self.batch = BoardBatch(boards)
        self.rollouts = rollouts
        self.rules = rules
        self.flags = np.array(rules.as_flags(), dtype=np.int64)
        n = len(self.batch) * rollouts
        self.n_games = n
        self.bidx = np.repeat(np.arange(len(self.batch), dtype=np.int64), rollouts)
        self.ST = np.zeros((n, NST), dtype=np.int64)
        self.FL = np.zeros((n, R.ROWS, R.COLS), dtype=np.int64)
        self.alive = np.zeros(n, dtype=np.bool_)
        self.plans = np.zeros((n, MAX_SLOTS), dtype=np.int8)
        self.scores = np.zeros(n, dtype=np.int64)
        self._dummy = np.zeros(1, dtype=np.int64)
        self._img = np.zeros((n, N_CHANNELS, R.ROWS, R.COLS), dtype=np.float32)
        self._vec = np.zeros((n, N_VECTOR), dtype=np.float32)
        self._mask = np.zeros((n, N_ACTIONS), dtype=np.bool_)

    def _args(self):
        b = self.batch
        return (self.bidx, b.tiles, b.bed_owner, b.sr, b.sc, b.prio, b.slot_base, self._dummy, self.flags)

    def reset(self) -> None:
        bidx, *rest = self._args()
        _reset_all(self.ST, self.FL, bidx, *rest, self.alive, self.plans)
        self.scores[:] = 0
        # games that never need a decision (nothing is ever destroyed) end immediately with their score
        done0 = ~self.alive
        for g in np.nonzero(done0)[0]:
            self.scores[g] = int(sum(self.ST[g, S_POWER + i] for i in range(3) if self.ST[g, S_STATUS + i] != ST_GONE))

    def any_alive(self) -> bool:
        return bool(self.alive.any())

    def observe(self):
        """Return (img, vec, mask) arrays for ALL games (rows of finished games are stale; use ``self.alive``)."""
        bidx, tiles, bed_owner, *_ , flags = self._args()
        _observe(self.ST, self.FL, bidx, tiles, bed_owner, flags, self.alive, self._img, self._vec, self._mask)
        return self._img, self._vec, self._mask

    def step(self, actions: np.ndarray) -> None:
        _step_all(self.ST, self.FL, *self._args()[:-1], self.flags, np.ascontiguousarray(actions, dtype=np.int64),
                  self.alive, self.plans, self.scores)

    def pending_slots(self) -> np.ndarray:
        """Slot index of the floor each live game is waiting for a command on (-1 for finished games)."""
        out = np.full(self.n_games, -1, dtype=np.int64)
        live = np.nonzero(self.alive)[0]
        out[live] = self.ST[live, S_EVSLOT + self.ST[live, S_EVPTR]]
        return out

    def spent(self) -> np.ndarray:
        return self.ST[:, S_SPENT].copy()
