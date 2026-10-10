"""
Exact "lazy decision" depth-first search used to re-optimise part of a plan.

Why it works
------------
A command only matters at the moment its floor is destroyed.  So instead of choosing a command for every
slot up front, the search *plays the game* and, whenever a destroyed floor belongs to a **free** slot,
branches over the sensible commands for that slot.  Slots that are not free keep the command from the base
plan.  With every slot free this is the exact optimum for the whole game; with only the slots of one cat free
it is the exact best way to re-route that cat while the others stay put ("cat-neighbourhood descent"), which
is a far bigger move than anything a mutation-based search can make in one step.

Pruning (all exact, apart from the negligible chance of a 64-bit hash collision):
* a direction command equal to the cat's current direction is the same as "no command" but costs money;
* Stomp on a floor that is already gone is the same as "no command";
* a transposition table remembers every game state already explored together with the least money spent
  to reach it; reaching it again having spent at least as much is pointless.
The node cap keeps worst cases bounded; the result then is "best found", not proven optimal, and the function
says so.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numba import njit

from . import rules as R
from .board import Board
from .fastsim import (NST, S_DIR, S_EVC, S_EVCAT, S_EVR, S_EVSLOT, S_NEV, S_EVPTR, S_SPENT,
                      S_TURN, _COST, all_done, apply_event, init_state, make_scratch, move_phase,
                      total_score)

MAX_DEPTH = 80                      # decision points on one root-to-leaf path (far more than a game ever needs)
TABLE_BITS = 21


@njit(cache=True)
def _hash_state(st, fl):
    """64-bit FNV-style hash of everything that determines the future except the money already spent."""
    h = np.uint64(1469598103934665603)
    prime = np.uint64(1099511628211)
    for k in range(NST):
        if k == S_SPENT:
            continue
        h = (h ^ np.uint64(st[k] + 1000003)) * prime
        h ^= h >> np.uint64(29)
    for r in range(fl.shape[0]):
        for c in range(fl.shape[1]):
            h = (h ^ np.uint64(fl[r, c] + 7)) * prime
            h ^= h >> np.uint64(31)
    return h | np.uint64(1)         # never 0, so 0 can mean "empty"


@njit(cache=True)
def _new_state():
    return np.zeros(NST, dtype=np.int64), np.zeros((R.ROWS, R.COLS), dtype=np.int64)


@njit(cache=True)
def slot_owners(plan, pack, flags, owner):
    """For every slot whose floor gets destroyed while playing `plan`, record which cat destroyed it (else -1)."""
    st, fl = _new_state()
    owner[:] = -1
    init_state(st, fl, pack, flags)
    for _ in range(flags[3]):
        if all_done(st):
            break
        move_phase(st, fl, pack, flags)
        for e in range(st[S_NEV]):
            slot = st[S_EVSLOT + e]
            owner[slot] = st[S_EVCAT + e]
            apply_event(st, fl, pack, flags, e, plan[slot])


@njit(cache=True)
def lazy_dfs(base, free, pack, flags, node_cap, out_plan, table_keys, table_spent):
    """Best plan reachable by re-deciding the `free` slots of `base` lazily.

    Returns (best_score, nodes_visited, finished) where finished is False if the node cap stopped the search.
    `out_plan` receives the best plan found (free slots that never fire are zeroed).
    """
    budget = flags[4]
    turns = flags[3]
    mask = table_keys.shape[0] - 1
    st, fl = _new_state()
    init_state(st, fl, pack, flags)

    stack_st = np.zeros((MAX_DEPTH, NST), dtype=np.int64)
    stack_fl = np.zeros((MAX_DEPTH, fl.shape[0], fl.shape[1]), dtype=np.int64)
    stack_slot = np.zeros(MAX_DEPTH, dtype=np.int64)
    stack_next = np.zeros(MAX_DEPTH, dtype=np.int64)         # next command to try at that decision
    dec = base.copy()
    for s in range(dec.shape[0]):
        if free[s]:
            dec[s] = 0
    best = -1
    nodes = 0
    depth = 0
    finished = True
    # `fresh` means: we are at a node that still has to be advanced; otherwise we are backtracking.
    backtrack = False
    while True:
        if not backtrack:
            # ---- advance the current state until it needs a decision or the game ends -------------------
            need_branch = False
            dead = False
            while True:
                while st[S_EVPTR] < st[S_NEV]:
                    e = st[S_EVPTR]
                    slot = st[S_EVSLOT + e]
                    if free[slot]:
                        need_branch = True
                        break
                    cmd = dec[slot]
                    if st[S_SPENT] + _COST[cmd] > budget:
                        dead = True
                        break
                    apply_event(st, fl, pack, flags, e, cmd)
                    st[S_EVPTR] = e + 1
                if need_branch or dead:
                    break
                if all_done(st) or st[S_TURN] >= turns:
                    break
                move_phase(st, fl, pack, flags)
            nodes += 1
            if dead:
                backtrack = True
                continue
            if not need_branch:
                score = total_score(st)
                if score > best:
                    best = score
                    out_plan[:] = dec
                backtrack = True
                continue
            # transposition check at decision points
            h = _hash_state(st, fl)
            idx = np.int64(h & np.uint64(mask))
            prev = table_keys[idx]
            if prev == h and table_spent[idx] <= st[S_SPENT]:
                backtrack = True
                continue
            table_keys[idx] = h
            table_spent[idx] = st[S_SPENT]
            if nodes >= node_cap:
                finished = False
                backtrack = True
                continue
            if depth >= MAX_DEPTH:
                backtrack = True
                continue
            stack_st[depth, :] = st
            stack_fl[depth, :, :] = fl
            stack_slot[depth] = st[S_EVSLOT + st[S_EVPTR]]
            stack_next[depth] = 0
            depth += 1
        # ---- backtrack / try the next command of the top decision ---------------------------------------
        backtrack = False
        found = False
        while depth > 0:
            d = depth - 1
            k = stack_next[d]
            if k > 6:
                dec[stack_slot[d]] = 0
                depth -= 1
                continue
            stack_next[d] = k + 1
            # try the base plan's own command first so a good incumbent exists after the very first dive
            b0 = base[stack_slot[d]]
            if k == 0:
                cmd = b0
            elif k - 1 < b0:
                cmd = k - 1
            else:
                cmd = k
            st[:] = stack_st[d]
            fl[:, :] = stack_fl[d]
            e = st[S_EVPTR]
            slot = stack_slot[d]
            cat = st[S_EVCAT + e]
            # prune dominated commands
            if cmd >= 1 and cmd <= 4 and st[S_DIR + cat] == cmd - 1:
                continue
            if cmd == R.CMD_STOMP:
                # Stomp only matters if a floor remains on that tile after this one is destroyed
                if fl[st[S_EVR + e], st[S_EVC + e]] <= 0:
                    continue
            if st[S_SPENT] + _COST[cmd] > budget:
                continue
            dec[slot] = cmd
            apply_event(st, fl, pack, flags, e, cmd)
            st[S_EVPTR] = e + 1
            found = True
            break
        if not found:
            break
        if nodes >= node_cap:
            finished = False
            break
    return best, nodes, finished


def new_table(bits: int = TABLE_BITS) -> Tuple[np.ndarray, np.ndarray]:
    return np.zeros(1 << bits, dtype=np.uint64), np.zeros(1 << bits, dtype=np.int64)


# GIL-free copies, so several boards can be re-optimised in parallel threads (used by the RL trainer)
lazy_dfs_nogil = njit(nogil=True, cache=True)(lazy_dfs.py_func)
slot_owners_nogil = njit(nogil=True, cache=True)(slot_owners.py_func)


def reoptimise(board: Board, cmds: np.ndarray, free: np.ndarray, rules: R.Rules = R.Rules(),
               node_cap: int = 2_000_000, solo: int = -1, table_bits: int = TABLE_BITS,
               nogil: bool = False) -> Tuple[int, np.ndarray, int, bool]:
    """Python front-end: returns (score, plan_cmds, nodes, finished)."""
    pack = board.numba_pack()
    flags = np.array(rules.as_flags(), dtype=np.int64)
    flags[5] = solo
    keys, spent = new_table(table_bits)
    out = np.zeros_like(cmds, dtype=np.int8)
    base = np.ascontiguousarray(cmds, dtype=np.int8)
    fn = lazy_dfs_nogil if nogil else lazy_dfs
    score, nodes, finished = fn(base, np.ascontiguousarray(free, dtype=np.uint8), pack, flags, node_cap,
                                      out, keys, spent)
    return int(score), out, int(nodes), bool(finished)


def descend(board: Board, cmds: np.ndarray, rules: R.Rules = R.Rules(), node_cap: int = 1_500_000,
            max_rounds: int = 8, table_bits: int = TABLE_BITS, nogil: bool = False) -> Tuple[np.ndarray, int]:
    """Cat-neighbourhood descent: repeatedly re-route one cat (then two) exactly while the rest stay put.

    Returns (plan_cmds, score).  Every accepted step strictly improves the score, so it terminates.
    """
    from .fastsim import FastSim
    sim = FastSim(board, rules)
    pack, flags = sim.pack, sim.flags
    cur = np.ascontiguousarray(cmds, dtype=np.int8)
    score = sim.evaluate(cur)[0]
    groups = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]
    for _ in range(max_rounds):
        owner = np.full(board.n_slots, -1, dtype=np.int64)
        (slot_owners_nogil if nogil else slot_owners)(cur, pack, flags, owner)
        for grp in groups:
            free = np.isin(owner, grp).astype(np.uint8)
            new_score, cand, _, _ = reoptimise(board, cur, free, rules, node_cap, table_bits=table_bits, nogil=nogil)
            if new_score > score and sim.evaluate(cand)[0] == new_score:
                cur, score = cand, new_score
                break
        else:
            break
    return canonical(board, cur, rules), int(score)


def canonical(board: Board, cmds: np.ndarray, rules: R.Rules = R.Rules()) -> np.ndarray:
    """Erase commands on floors that never get destroyed.

    They do nothing in the game, but they would still count against the $200 you may place, so a plan that
    keeps them can look over budget.  Always canonicalise before showing or storing a plan.
    """
    from .fastsim import FastSim
    sim = FastSim(board, rules)
    _, _, used, _, _ = sim.evaluate(np.ascontiguousarray(cmds, dtype=np.int8))
    out = np.array(cmds, dtype=np.int8, copy=True)
    out[used == 0] = 0
    return out
