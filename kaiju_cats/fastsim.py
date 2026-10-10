"""
numba-compiled simulator and search kernels.

``simulate_fast`` plays exactly the same game as :mod:`kaiju_cats.simulator` (the tests check this on
thousands of random plans) but about 100x faster, which is what makes a Python search practical:
the parallel-tempering search below evaluates millions of plans per minute.

Design notes
------------
* A plan is an ``int8`` vector, one command per slot (0 = none).
* After every accepted move the plan is *canonicalised*: commands on floors that never got destroyed are
  erased.  They cost nothing and change nothing, so keeping them around would only blur the search.
* The search state lives in plain numpy arrays so it survives between calls; that lets the Python driver
  run a kernel for a fraction of a second, look at the clock, and decide whether the search has plateaued.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from . import rules as R

# Global constants are frozen into the compiled code by numba.
_COST = np.array(R.CMD_COST, dtype=np.int64)
_DR = np.array(R.DR, dtype=np.int64)
_DC = np.array(R.DC, dtype=np.int64)
_FLOOR_VALUE = np.zeros(9, dtype=np.int64)
_FLOOR_VALUE[R.T_LOW] = R.FLOOR_VALUE[R.T_LOW]
_FLOOR_VALUE[R.T_HIGH] = R.FLOOR_VALUE[R.T_HIGH]

ST_ACTIVE, ST_STUCK, ST_STOMP, ST_BED, ST_GONE = 0, 1, 2, 3, 4


# ---- game state layout: ONE int64 vector, so the DFS can copy a whole state with a slice assignment -------
S_ROW, S_COL, S_DIR, S_STATUS, S_POWER = 0, 3, 6, 9, 12       # 3 entries each (one per cat)
S_ARRIVED, S_SPENT, S_TURN = 15, 16, 17
S_ORDER = 18                                                  # 3 entries: cats in bed-arrival order (-1 = none yet)
S_NEV, S_EVPTR = 21, 22                                       # events produced this turn / how many are handled
S_EVCAT, S_EVSLOT, S_EVR, S_EVC = 23, 26, 29, 32              # 3 entries each (at most one event per cat per turn)
NST = 35


def make_scratch():
    """Preallocated work arrays (state vector, floor grid) for the kernels - one set per thread/process.

    Passing them in avoids tiny allocations that would otherwise cost more than the simulation itself.
    """
    return np.zeros(NST, dtype=np.int64), np.zeros((R.ROWS, R.COLS), dtype=np.int64)


@njit(cache=True)
def init_state(st, fl, pack, flags):
    """Reset `st`/`fl` to the start of a game.  flags[5] >= 0 keeps only that cat (single-cat relaxation)."""
    tiles, bed_owner, sr, sc, prio, slot_base, slot_r, slot_c = pack
    solo = flags[5]
    for k in range(NST):
        st[k] = 0
    for i in range(3):
        st[S_ROW + i] = sr[i]
        st[S_COL + i] = sc[i]
        st[S_DIR + i] = R.EAST                                  # everybody starts facing East
        st[S_STATUS + i] = ST_ACTIVE if (solo < 0 or solo == i) else ST_GONE
        st[S_ORDER + i] = -1
    # in the single-cat relaxation the cat is treated as the LAST arrival (x5)
    st[S_ARRIVED] = 2 if solo >= 0 else 0
    for r in range(tiles.shape[0]):
        for c in range(tiles.shape[1]):
            t = tiles[r, c]
            fl[r, c] = 2 if (t == R.T_LOW or t == R.T_HIGH) else (1 if t == R.T_PLANT else 0)


@njit(cache=True)
def all_done(st):
    return st[S_STATUS] >= ST_BED and st[S_STATUS + 1] >= ST_BED and st[S_STATUS + 2] >= ST_BED


@njit(cache=True)
def _queue_event(st, fl, tiles, slot_base, cat, r, c):
    """Destroy the top floor at (r, c) and queue the (cat, slot) so its command can fire later this turn."""
    total = 1 if tiles[r, c] == R.T_PLANT else 2
    n = st[S_NEV]
    st[S_EVCAT + n] = cat
    st[S_EVSLOT + n] = slot_base[r, c] + (total - fl[r, c])
    st[S_EVR + n] = r
    st[S_EVC + n] = c
    fl[r, c] -= 1
    st[S_NEV] = n + 1


@njit(cache=True)
def move_phase(st, fl, pack, flags):
    """Advance one turn of movement, fights, landings and bed arrivals; queue the floor-destroy events."""
    tiles, bed_owner, sr, sc, prio, slot_base, slot_r, slot_c = pack
    bounce_breaks = flags[0] == 1
    edge_costs_turn = flags[1] == 1
    nrow, ncol = tiles.shape
    st[S_TURN] += 1
    st[S_NEV] = 0
    st[S_EVPTR] = 0
    # per-cat scratch kept in small local tuples (no allocation): target tile, moving flag, bounced flag, alive flag
    tr0, tr1, tr2 = st[S_ROW], st[S_ROW + 1], st[S_ROW + 2]
    tc0, tc1, tc2 = st[S_COL], st[S_COL + 1], st[S_COL + 2]
    tr = np.array([tr0, tr1, tr2])
    tc = np.array([tc0, tc1, tc2])
    moving = np.zeros(3, dtype=np.bool_)
    bounced = np.zeros(3, dtype=np.bool_)
    alive = np.zeros(3, dtype=np.bool_)
    for i in range(3):
        alive[i] = st[S_STATUS + i] < ST_BED
    # ---- 1. intentions --------------------------------------------------------------------
    for i in range(3):
        if not alive[i]:
            continue
        if st[S_STATUS + i] == ST_STUCK:
            st[S_STATUS + i] = ST_ACTIVE
            continue
        if st[S_STATUS + i] == ST_STOMP:
            continue
        d = st[S_DIR + i]
        nr = st[S_ROW + i] + _DR[d]
        nc = st[S_COL + i] + _DC[d]
        blocked = False
        edge = False
        if nr < 0 or nr >= nrow or nc < 0 or nc >= ncol:
            blocked = True
            edge = True
        else:
            t = tiles[nr, nc]
            if t == R.T_NONE:
                blocked = True
                edge = True
            elif t == R.T_BOULDER:
                blocked = True
            elif t == R.T_BED and bed_owner[nr, nc] != i:
                blocked = True
        if blocked:
            st[S_DIR + i] = (d + 2) % 4
            bounced[i] = True
            if edge and not edge_costs_turn:
                d2 = st[S_DIR + i]
                nr = st[S_ROW + i] + _DR[d2]
                nc = st[S_COL + i] + _DC[d2]
                ok = not (nr < 0 or nr >= nrow or nc < 0 or nc >= ncol)
                if ok:
                    t = tiles[nr, nc]
                    if t == R.T_NONE or t == R.T_BOULDER or (t == R.T_BED and bed_owner[nr, nc] != i):
                        ok = False
                if ok:
                    tr[i] = nr
                    tc[i] = nc
                    moving[i] = True
        else:
            tr[i] = nr
            tc[i] = nc
            moving[i] = True
    # ---- 2. fights (powers are read before anything moves) ---------------------------------------
    lost = np.zeros(3, dtype=np.bool_)
    for i in range(3):
        if not alive[i]:
            continue
        for j in range(3):
            if j == i or not alive[j]:
                continue
            if tr[i] == tr[j] and tc[i] == tc[j]:
                if st[S_POWER + j] > st[S_POWER + i] or (st[S_POWER + j] == st[S_POWER + i] and prio[j] > prio[i]):
                    lost[i] = True
    for i in range(3):
        if lost[i]:
            st[S_STATUS + i] = ST_GONE
            st[S_POWER + i] = 0
            alive[i] = False
    # ---- 3. move and land -----------------------------------------------------------------------
    arr = np.zeros(3, dtype=np.int64)
    n_arr = 0
    for i in range(3):
        if not alive[i]:
            continue
        r0, c0 = st[S_ROW + i], st[S_COL + i]
        if moving[i]:
            st[S_ROW + i] = tr[i]
            st[S_COL + i] = tc[i]
            t = tiles[tr[i], tc[i]]
            if t == R.T_BED:
                arr[n_arr] = i
                n_arr += 1
            elif t == R.T_MUD:
                st[S_STATUS + i] = ST_STUCK
            elif t == R.T_SPIKE:
                st[S_POWER + i] = st[S_POWER + i] // 2
            elif (t == R.T_LOW or t == R.T_HIGH or t == R.T_PLANT) and fl[tr[i], tc[i]] > 0:
                _queue_event(st, fl, tiles, slot_base, i, tr[i], tc[i])
        elif st[S_STATUS + i] == ST_STOMP:
            if fl[r0, c0] > 0:
                _queue_event(st, fl, tiles, slot_base, i, r0, c0)
            else:
                st[S_STATUS + i] = ST_ACTIVE
        elif bounced[i] and bounce_breaks:
            t = tiles[r0, c0]
            if (t == R.T_LOW or t == R.T_HIGH or t == R.T_PLANT) and fl[r0, c0] > 0:
                _queue_event(st, fl, tiles, slot_base, i, r0, c0)
    # ---- 4. bed arrivals (lowest power first; ties -> lower bed first) -----------------------------
    for a in range(n_arr):
        for b in range(a + 1, n_arr):
            x = arr[a]
            y = arr[b]
            if st[S_POWER + y] < st[S_POWER + x] or (st[S_POWER + y] == st[S_POWER + x] and prio[y] < prio[x]):
                arr[a] = y
                arr[b] = x
    for a in range(n_arr):
        i = arr[a]
        rank = st[S_ARRIVED]
        if rank == 0:
            st[S_POWER + i] += R.ARRIVAL_ADD_FIRST
        elif rank == 1:
            st[S_POWER + i] *= R.ARRIVAL_MULT_SECOND
        else:
            st[S_POWER + i] *= R.ARRIVAL_MULT_THIRD
        if rank < 3:
            st[S_ORDER + rank] = i
        st[S_ARRIVED] = rank + 1
        st[S_STATUS + i] = ST_BED


@njit(cache=True)
def apply_event(st, fl, pack, flags, e, cmd):
    """Pay out the floor behind queued event `e`, then fire `cmd`."""
    tiles = pack[0]
    i = st[S_EVCAT + e]
    r, c = st[S_EVR + e], st[S_EVC + e]
    pp_order = flags[2]
    if st[S_STATUS + i] == ST_STOMP:
        st[S_STATUS + i] = ST_ACTIVE
    st[S_SPENT] += _COST[cmd]
    t = tiles[r, c]
    if t == R.T_PLANT:
        if pp_order == 0:                                       # double, then add 1000
            st[S_POWER + i] *= 2
            if cmd == R.CMD_POWER_UP:
                st[S_POWER + i] += R.POWER_UP_BONUS
        else:                                                   # add 1000, then double
            if cmd == R.CMD_POWER_UP:
                st[S_POWER + i] += R.POWER_UP_BONUS
            st[S_POWER + i] *= 2
    else:
        st[S_POWER + i] += _FLOOR_VALUE[t]
        if cmd == R.CMD_POWER_UP:
            st[S_POWER + i] += R.POWER_UP_BONUS
    if cmd >= 1 and cmd <= 4:
        st[S_DIR + i] = cmd - 1
    elif cmd == R.CMD_STOMP and fl[r, c] > 0:
        st[S_STATUS + i] = ST_STOMP


@njit(cache=True)
def total_score(st):
    total = 0
    for i in range(3):
        if st[S_STATUS + i] != ST_GONE:
            total += st[S_POWER + i]
    return total


@njit(cache=True)
def sim_kernel(plan, pack, flags, used, order, cat_power, scratch):
    """Play `plan`.  Returns (score, money_spent).

    Side outputs (written into the arrays you pass in):
      used[slot]    1 if that slot's floor was destroyed (so its command was consulted)
      order[k]      the cat that entered its bed k-th (-1 if fewer than k+1 cats arrived)
      cat_power[i]  final power of cat i (0 if it vanished)
    `scratch` is the tuple from :func:`make_scratch`.
    """
    st, fl = scratch
    init_state(st, fl, pack, flags)
    for i in range(used.shape[0]):
        used[i] = 0
    turns = flags[3]
    for _ in range(turns):
        if all_done(st):
            break
        move_phase(st, fl, pack, flags)
        for e in range(st[S_NEV]):
            slot = st[S_EVSLOT + e]
            used[slot] = 1
            apply_event(st, fl, pack, flags, e, plan[slot])
    for k in range(3):
        order[k] = st[S_ORDER + k]
    for i in range(3):
        cat_power[i] = 0 if st[S_STATUS + i] == ST_GONE else st[S_POWER + i]
    return total_score(st), st[S_SPENT]


class FastSim:
    """Small convenience wrapper that owns the scratch arrays for repeated evaluations."""

    def __init__(self, board, rules: R.Rules = R.Rules()):
        self.board, self.rules = board, rules
        self.pack = board.numba_pack()
        self.flags = np.array(rules.as_flags(), dtype=np.int64)
        self._used = np.zeros(board.n_slots, dtype=np.uint8)
        self._order = np.zeros(3, dtype=np.int64)
        self._power = np.zeros(3, dtype=np.int64)
        self._scratch = make_scratch()

    def evaluate(self, cmds):
        """Return (score, money_spent, used_slots_mask, arrival_order, cat_power) for a command vector."""
        score, spent = sim_kernel(np.ascontiguousarray(cmds, dtype=np.int8), self.pack, self.flags,
                                  self._used, self._order, self._power, self._scratch)
        return int(score), int(spent), self._used.copy(), self._order.copy(), self._power.copy()


# =============================================================================================
# Parallel tempering
# =============================================================================================
@njit(cache=True)
def _rng_next(rng):
    """xorshift64 - tiny, fast and good enough for a stochastic search."""
    x = rng[0]
    x ^= x << np.uint64(13)
    x ^= x >> np.uint64(7)
    x ^= x << np.uint64(17)
    rng[0] = x
    return x


@njit(cache=True)
def _rand_int(rng, n):
    return np.int64(_rng_next(rng) % np.uint64(n))


@njit(cache=True)
def _rand01(rng):
    return (_rng_next(rng) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def _fitness(score, order, req, order_penalty):
    """Score, minus a heavy penalty for every position where the bed-arrival order differs from `req`.

    Conditioning the search on an arrival order is the single most useful trick in this solver: the
    arrival bonus (+2000 / x3 / x5) creates six very different "basins" and a plain search tends to get
    stuck in whichever it finds first.
    """
    if req[0] < 0:
        return float(score)
    viol = 0
    for k in range(3):
        if order[k] >= 0 and order[k] != req[k]:
            viol += 1
    return float(score) - order_penalty * viol


@njit(cache=True)
def _pick_used(rng, used_row, n_used):
    """Pick uniformly among slots whose floor was destroyed."""
    k = _rand_int(rng, n_used)
    for s in range(used_row.shape[0]):
        if used_row[s] == 1:
            if k == 0:
                return s
            k -= 1
    return 0


@njit(cache=True)
def _mutate(rng, plan_row, used_row, out, p_swap):
    """Propose a neighbouring plan in `out`.

    * with probability `p_swap`: swap or move a command between two slots (re-routes without changing the budget)
    * otherwise: change 1-3 slots to a random different command, preferring slots that currently matter
    """
    n_slots = plan_row.shape[0]
    for s in range(n_slots):
        out[s] = plan_row[s]
    n_used = 0
    for s in range(n_slots):
        n_used += used_row[s]
    if n_used > 0 and _rand_int(rng, 100) < p_swap:
        a = _pick_used(rng, used_row, n_used)
        if _rand_int(rng, 100) < 60:
            b = _pick_used(rng, used_row, n_used)
        else:
            b = _rand_int(rng, n_slots)
        if a != b:
            if _rng_next(rng) & np.uint64(1):
                tmp = out[a]
                out[a] = out[b]
                out[b] = tmp
            else:
                out[b] = out[a]
                out[a] = 0
    else:
        k = 1
        if _rand_int(rng, 100) < 35:
            k += 1
        if _rand_int(rng, 100) < 12:
            k += 1
        for _ in range(k):
            if n_used > 0 and _rand_int(rng, 100) < 85:
                s = _pick_used(rng, used_row, n_used)
            else:
                s = _rand_int(rng, n_slots)
            nv = _rand_int(rng, R.N_COMMANDS)
            while nv == out[s]:
                nv = _rand_int(rng, R.N_COMMANDS)
            out[s] = nv


@njit(cache=True)
def pt_chunk(plans, used, scores, costs, fits, temps, best_plan, best_info, rng, req, pack, flags,
             n_sweeps, steps_per_sweep, order_penalty, p_swap, sweep_counter, scratch):
    """Run `n_sweeps` sweeps of parallel tempering.

    Every sweep each replica does `steps_per_sweep` Metropolis steps at its own temperature, then
    neighbouring replicas are offered a configuration swap (hot replicas explore, cold ones refine).

    best_info = [best_score, best_cost, last_improvement_sweep]
    sweep_counter is a 1-element array holding the total number of sweeps done so far.
    """
    n_rep, n_slots = plans.shape
    budget = flags[4]
    cand = np.zeros(n_slots, dtype=np.int8)
    cand_used = np.zeros(n_slots, dtype=np.uint8)
    order = np.zeros(3, dtype=np.int64)
    power = np.zeros(3, dtype=np.int64)
    tmp_plan = np.zeros(n_slots, dtype=np.int8)
    tmp_used = np.zeros(n_slots, dtype=np.uint8)
    for _ in range(n_sweeps):
        for k in range(n_rep):
            for _step in range(steps_per_sweep):
                _mutate(rng, plans[k], used[k], cand, p_swap)
                score, spent = sim_kernel(cand, pack, flags, cand_used, order, power, scratch)
                if spent > budget:
                    continue
                nf = _fitness(score, order, req, order_penalty)
                d = nf - fits[k]
                if d >= 0 or _rand01(rng) < np.exp(d / temps[k]):
                    for s in range(n_slots):                    # canonicalise: drop commands that never fired
                        plans[k, s] = cand[s] if cand_used[s] == 1 else 0
                        used[k, s] = cand_used[s]
                    scores[k] = score
                    costs[k] = spent
                    fits[k] = nf
                    if score > best_info[0]:
                        best_info[0] = score
                        best_info[1] = spent
                        best_info[2] = sweep_counter[0]
                        for s in range(n_slots):
                            best_plan[s] = plans[k, s]
        # replica exchange between neighbouring temperatures (alternate even/odd pairs)
        for k in range(sweep_counter[0] & 1, n_rep - 1, 2):
            d = (fits[k + 1] - fits[k]) * (1.0 / temps[k] - 1.0 / temps[k + 1])
            if d >= 0 or _rand01(rng) < np.exp(d):
                for s in range(n_slots):
                    tmp_plan[s] = plans[k, s]
                    tmp_used[s] = used[k, s]
                    plans[k, s] = plans[k + 1, s]
                    used[k, s] = used[k + 1, s]
                    plans[k + 1, s] = tmp_plan[s]
                    used[k + 1, s] = tmp_used[s]
                ts, tco, tf = scores[k], costs[k], fits[k]
                scores[k], costs[k], fits[k] = scores[k + 1], costs[k + 1], fits[k + 1]
                scores[k + 1], costs[k + 1], fits[k + 1] = ts, tco, tf
        sweep_counter[0] += 1


# =============================================================================================
# Exhaustive local search (polish)
# =============================================================================================
@njit(cache=True)
def polish_kernel(plan, pack, flags, depth, scratch):
    """Hill-climb with EXHAUSTIVE k-opt moves, k = 1..depth (depth 1, 2 or 3).

    A k-opt move changes the command on k slots at once.  We try all of them and take the best
    strictly-improving one, repeating until nothing improves.  Pairs catch things like "move this Power Up
    one building later while re-aiming the turn", which single changes cannot see.
    Returns (plan, score, cost).
    """
    n_slots = plan.shape[0]
    budget = flags[4]
    used = np.zeros(n_slots, dtype=np.uint8)
    order = np.zeros(3, dtype=np.int64)
    power = np.zeros(3, dtype=np.int64)
    cur = plan.copy()
    best_score, best_cost = sim_kernel(cur, pack, flags, used, order, power, scratch)
    for s in range(n_slots):
        if used[s] == 0:
            cur[s] = 0
    best_score, best_cost = sim_kernel(cur, pack, flags, used, order, power, scratch)
    trial = cur.copy()
    while True:
        found = False
        move_score = best_score
        move = np.zeros(3, dtype=np.int64)      # slot indices of the best move found in this pass
        move_val = np.zeros(3, dtype=np.int64)
        move_k = 0
        for a in range(n_slots):
            for va in range(R.N_COMMANDS):
                if va == cur[a]:
                    continue
                trial[a] = va
                sc, co = sim_kernel(trial, pack, flags, used, order, power, scratch)
                if co <= budget and sc > move_score:
                    move_score, move_k, found = sc, 1, True
                    move[0], move_val[0] = a, va
                if depth >= 2:
                    for b in range(a + 1, n_slots):
                        for vb in range(R.N_COMMANDS):
                            if vb == cur[b]:
                                continue
                            trial[b] = vb
                            sc, co = sim_kernel(trial, pack, flags, used, order, power, scratch)
                            if co <= budget and sc > move_score:
                                move_score, move_k, found = sc, 2, True
                                move[0], move_val[0] = a, va
                                move[1], move_val[1] = b, vb
                            if depth >= 3:
                                for c in range(b + 1, n_slots):
                                    for vc in range(R.N_COMMANDS):
                                        if vc == cur[c]:
                                            continue
                                        trial[c] = vc
                                        sc, co = sim_kernel(trial, pack, flags, used, order, power, scratch)
                                        if co <= budget and sc > move_score:
                                            move_score, move_k, found = sc, 3, True
                                            move[0], move_val[0] = a, va
                                            move[1], move_val[1] = b, vb
                                            move[2], move_val[2] = c, vc
                                    trial[c] = cur[c]
                            trial[b] = cur[b]
                trial[a] = cur[a]
        if not found:
            break
        for m in range(move_k):
            cur[move[m]] = move_val[m]
        best_score, best_cost = sim_kernel(cur, pack, flags, used, order, power, scratch)
        for s in range(n_slots):                    # canonicalise
            if used[s] == 0:
                cur[s] = 0
        best_score, best_cost = sim_kernel(cur, pack, flags, used, order, power, scratch)
        trial[:] = cur
    return cur, best_score, best_cost


# A copy of polish_kernel that releases the GIL, so several polishes can run in parallel threads
# (used by the expert-iteration trainer; the search solver uses processes instead).
polish_nogil = njit(nogil=True, cache=True)(polish_kernel.py_func)
