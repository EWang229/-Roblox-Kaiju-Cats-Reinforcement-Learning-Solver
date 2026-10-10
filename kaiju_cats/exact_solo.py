"""
Exact single-cat route search.

Take ONE cat, remove the others (no fights, no arrival-order bookkeeping) and ask: what is the highest power this cat
can reach when it enters its bed as the *last* cat (x5), for each amount of money spent?  That is a much smaller
problem than the full game - one cat makes at most one decision per turn - so we can solve it *exactly* with a
depth-first search over the commands, using a transposition table to throw away repeated or dominated situations.

Why bother?  In every board we solved, 85-95 percent of the final score sits on one "big" cat.  Knowing that cat's best
possible route (and its price) tells you what to build the rest of the plan around; it is also a strong upper bound on
what any plan can do with that cat as the last arrival.

"Lazy decisions" - the trick that makes this tractable: a command only matters at the instant its floor is destroyed, so
we branch only when the cat actually destroys a floor, not over every floor of the board.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from . import rules as R

I_ROW, I_COL, I_DIR, I_STATUS, I_POWER, I_SPENT, I_TURN, I_EV_SLOT, I_EV_R, I_EV_C, I_ARRIVED = range(11)
N_FIELDS = 11
EVENT, DONE = 1, 2

_COST = np.array(R.CMD_COST, dtype=np.int64)
_VALUE = np.zeros(9, dtype=np.int64)
_VALUE[R.T_LOW], _VALUE[R.T_HIGH] = R.FLOOR_VALUE[R.T_LOW], R.FLOOR_VALUE[R.T_HIGH]
_DR = np.array(R.DR, dtype=np.int64)
_DC = np.array(R.DC, dtype=np.int64)


@njit(cache=True)
def _event(st, fl, tiles, slot_base, r, c):
    """Destroy the top floor at (r, c) and remember which slot that was."""
    total = 1 if tiles[r, c] == R.T_PLANT else 2
    st[I_EV_SLOT] = slot_base[r, c] + (total - fl[r, c])
    fl[r, c] -= 1
    st[I_EV_R] = r
    st[I_EV_C] = c
    return EVENT


@njit(cache=True)
def _advance(st, fl, tiles, bed_owner, cat, slot_base, flags):
    """Run turns until the cat destroys a floor (returns EVENT) or the game ends (returns DONE)."""
    bounce_breaks = flags[0] == 1
    edge_costs_turn = flags[1] == 1
    turns = flags[3]
    nrow, ncol = tiles.shape
    while st[I_TURN] < turns:
        st[I_TURN] += 1
        if st[I_STATUS] == 1:                       # stuck in mud
            st[I_STATUS] = 0
            continue
        r, c = st[I_ROW], st[I_COL]
        if st[I_STATUS] == 2:                       # stomping: break the next floor of this building
            if fl[r, c] > 0:
                return _event(st, fl, tiles, slot_base, r, c)
            st[I_STATUS] = 0
            continue
        d = st[I_DIR]
        nr, nc = r + _DR[d], c + _DC[d]
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
            elif t == R.T_BOULDER or (t == R.T_BED and bed_owner[nr, nc] != cat):
                blocked = True
        if blocked:
            st[I_DIR] = (d + 2) % 4
            moved = False
            if edge and not edge_costs_turn:
                d2 = st[I_DIR]
                nr, nc = r + _DR[d2], c + _DC[d2]
                if 0 <= nr < nrow and 0 <= nc < ncol:
                    t = tiles[nr, nc]
                    if not (t == R.T_NONE or t == R.T_BOULDER or (t == R.T_BED and bed_owner[nr, nc] != cat)):
                        moved = True
            if not moved:
                if bounce_breaks and fl[r, c] > 0:
                    return _event(st, fl, tiles, slot_base, r, c)
                continue
        st[I_ROW], st[I_COL] = nr, nc
        t = tiles[nr, nc]
        if t == R.T_BED:
            st[I_ARRIVED] = 1                           # arrival bonus is applied by the caller (it depends on rank)
            return DONE
        if t == R.T_MUD:
            st[I_STATUS] = 1
        elif t == R.T_SPIKE:
            st[I_POWER] //= 2
        elif (t == R.T_LOW or t == R.T_HIGH or t == R.T_PLANT) and fl[nr, nc] > 0:
            return _event(st, fl, tiles, slot_base, nr, nc)
    return DONE


@njit(cache=True)
def _apply(st, fl, tiles, cmd, pp_order):
    """Pay out the destroyed floor and fire `cmd`."""
    r, c = st[I_EV_R], st[I_EV_C]
    if st[I_STATUS] == 2:
        st[I_STATUS] = 0
    st[I_SPENT] += _COST[cmd]
    t = tiles[r, c]
    if t == R.T_PLANT:
        if pp_order == 0:
            st[I_POWER] *= 2
            if cmd == R.CMD_POWER_UP:
                st[I_POWER] += R.POWER_UP_BONUS
        else:
            if cmd == R.CMD_POWER_UP:
                st[I_POWER] += R.POWER_UP_BONUS
            st[I_POWER] *= 2
    else:
        st[I_POWER] += _VALUE[t]
        if cmd == R.CMD_POWER_UP:
            st[I_POWER] += R.POWER_UP_BONUS
    if 1 <= cmd <= 4:
        st[I_DIR] = cmd - 1
    elif cmd == R.CMD_STOMP and fl[r, c] > 0:
        st[I_STATUS] = 2


@njit(cache=True)
def _state_hash(st, fl):
    """64-bit fingerprint of everything that influences the future except power and money."""
    h = np.uint64(1469598103934665603)
    prime = np.uint64(1099511628211)
    for k in (I_ROW, I_COL, I_DIR, I_STATUS, I_TURN):
        h = (h ^ np.uint64(st[k] + 1)) * prime
    for r in range(fl.shape[0]):
        for c in range(fl.shape[1]):
            h = (h ^ np.uint64(fl[r, c] + 1)) * prime
    h ^= h >> np.uint64(29)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(32)
    return h


@njit(cache=True)
def solo_dfs(pack, flags, cat, budget, memo_log2, max_nodes):
    """Exact search.  Returns (arrive_power, arrive_plan, stay_power, nodes, finished).

    arrive_power[k] = best power the cat can have the moment it enters its bed (BEFORE the arrival bonus), spending AT
                      MOST $10*k; -1 if it cannot reach its bed at all.
    arrive_plan[k]  = a command vector (one entry per slot) that achieves it.
    stay_power[k]   = best power if the cat never reaches its bed (only matters for degenerate boards).
    Multiply arrive_power by 5 / 3, or add 2000, for a last / second / first arrival.
    """
    tiles, bed_owner, sr, sc, prio, slot_base, slot_r, slot_c = pack
    pp_order = flags[2]
    n_slots = slot_r.shape[0]
    nrow, ncol = tiles.shape
    max_depth = flags[3] + 3
    S = np.zeros((max_depth + 1, N_FIELDS), dtype=np.int64)
    F = np.zeros((max_depth + 1, nrow, ncol), dtype=np.int64)
    opt = np.zeros(max_depth + 1, dtype=np.int64)
    dec_slot = np.zeros(max_depth + 1, dtype=np.int64)
    dec_cmd = np.zeros(max_depth + 1, dtype=np.int64)

    n_money = budget // 10 + 1
    best = np.full(n_money, -1, dtype=np.int64)
    stay = np.full(n_money, -1, dtype=np.int64)
    best_plan = np.zeros((n_money, n_slots), dtype=np.int8)

    memo_size = 1 << memo_log2
    memo_key = np.zeros(memo_size, dtype=np.uint64)
    memo_pow = np.zeros(memo_size, dtype=np.int64)
    memo_spent = np.zeros(memo_size, dtype=np.int64)
    mask = np.uint64(memo_size - 1)

    S[0, I_ROW], S[0, I_COL], S[0, I_DIR] = sr[cat], sc[cat], 1
    for r in range(nrow):
        for c in range(ncol):
            t = tiles[r, c]
            F[0, r, c] = 2 if (t == R.T_LOW or t == R.T_HIGH) else (1 if t == R.T_PLANT else 0)
    nodes = 0
    code = _advance(S[0], F[0], tiles, bed_owner, cat, slot_base, flags)
    if code == DONE:
        if S[0, I_ARRIVED] == 1:
            best[0] = S[0, I_POWER]
        else:
            stay[0] = S[0, I_POWER]
        return best, best_plan, stay, nodes, True

    d = 0
    opt[0] = 0
    finished = True
    while d >= 0:
        if nodes >= max_nodes:
            finished = False
            break
        # next affordable, non-redundant command for the event waiting at depth d
        c = opt[d]
        r_ev, c_ev = S[d, I_EV_R], S[d, I_EV_C]
        while c < R.N_COMMANDS:
            if S[d, I_SPENT] + _COST[c] > budget:
                c += 1
                continue
            if 1 <= c <= 4 and c - 1 == S[d, I_DIR]:      # "turn to where you already face" = a wasted $10
                c += 1
                continue
            if c == R.CMD_STOMP and F[d, r_ev, c_ev] <= 0:  # nothing left to stomp
                c += 1
                continue
            break
        if c >= R.N_COMMANDS:
            d -= 1
            continue
        opt[d] = c + 1
        for k in range(N_FIELDS):
            S[d + 1, k] = S[d, k]
        for r in range(nrow):
            for cc in range(ncol):
                F[d + 1, r, cc] = F[d, r, cc]
        dec_slot[d] = S[d, I_EV_SLOT]
        dec_cmd[d] = c
        _apply(S[d + 1], F[d + 1], tiles, c, pp_order)
        code = _advance(S[d + 1], F[d + 1], tiles, bed_owner, cat, slot_base, flags)
        nodes += 1
        if code == DONE:
            sp = S[d + 1, I_SPENT] // 10
            score = S[d + 1, I_POWER]
            if S[d + 1, I_ARRIVED] == 0:
                stay[sp] = max(stay[sp], score)
            elif score > best[sp]:
                best[sp] = score
                best_plan[sp, :] = 0
                for k in range(d + 1):
                    if dec_cmd[k] != 0:
                        best_plan[sp, dec_slot[k]] = dec_cmd[k]
            continue
        # transposition table: skip the node if an earlier visit had >= power for <= money in the same situation
        h = _state_hash(S[d + 1], F[d + 1])
        slot = np.int64(h & mask)
        hit = False
        for probe in range(4):
            idx = (slot + probe) & (memo_size - 1)
            if memo_key[idx] == h:
                if memo_pow[idx] >= S[d + 1, I_POWER] and memo_spent[idx] <= S[d + 1, I_SPENT]:
                    hit = True
                else:
                    memo_pow[idx] = max(memo_pow[idx], S[d + 1, I_POWER]) if S[d + 1, I_SPENT] >= memo_spent[idx] \
                        else S[d + 1, I_POWER]
                    memo_spent[idx] = min(memo_spent[idx], S[d + 1, I_SPENT])
                break
            if memo_key[idx] == 0:
                memo_key[idx] = h
                memo_pow[idx] = S[d + 1, I_POWER]
                memo_spent[idx] = S[d + 1, I_SPENT]
                break
        else:
            idx = slot                                  # table region full: overwrite (it is only a cache)
            memo_key[idx] = h
            memo_pow[idx] = S[d + 1, I_POWER]
            memo_spent[idx] = S[d + 1, I_SPENT]
        if hit:
            continue
        d += 1
        opt[d] = 0
    # make "at most $10k" monotone
    for k in range(1, n_money):
        if best[k - 1] > best[k]:
            best[k] = best[k - 1]
            best_plan[k, :] = best_plan[k - 1, :]
        stay[k] = max(stay[k], stay[k - 1])
    return best, best_plan, stay, nodes, finished


def best_solo_routes(board, cat: int, budget: int = R.BUDGET, rules: R.Rules = R.Rules(),
                     memo_log2: int = 24, max_nodes: int = 400_000_000):
    """Python wrapper around :func:`solo_dfs`: returns (arrive_power, arrive_plan, stay_power, nodes, finished)."""
    flags = np.array(rules.as_flags(), dtype=np.int64)
    flags[5] = cat
    return solo_dfs(board.numba_pack(), flags, cat, budget, memo_log2, max_nodes)
