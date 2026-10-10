"""
The solver: order-conditioned parallel tempering + exhaustive local polish, run in parallel with joblib.

Why this design (the short version, see docs/INSIGHTS.md for the long one)
-------------------------------------------------------------------------
Kaiju Cats is a deterministic puzzle: a plan (commands on floors) fully determines the final score, so there is
nothing to *learn* - the job is to search a rugged landscape of ~10^30 plans for the best one.  Three things made
the search reliable:

1. **A fast simulator** (numba) - millions of evaluations per minute.
2. **Parallel tempering** - replicas at different "temperatures" so cold ones refine good plans while hot ones
   keep exploring; neighbours swap configurations.
3. **Conditioning on the bed-arrival order.**  The score is dominated by the arrival bonus (+2000, x3, x5), and
   there are only six possible arrival orders.  Running a separate search per order stops the solver from
   falling into the first basin it finds - on one board plain search stalled at 166k while the best was 178k.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed

from . import rules as R
from .board import Board
from .lazy_dfs import descend
from .fastsim import FastSim, make_scratch, polish_kernel, pt_chunk, sim_kernel
from .plan import Plan
from .simulator import simulate


# =============================================================================================
# Configuration
# =============================================================================================
@dataclass(frozen=True)
class SolveConfig:
    """All the knobs of a solve.  Use :func:`SolveConfig.preset` for sensible bundles."""

    chains_per_order: int = 2         # independent parallel-tempering chains for every arrival order
    orders: Optional[Tuple[Tuple[int, ...], ...]] = None   # None -> all six orders
    min_seconds: float = 8.0          # a chain always runs at least this long ...
    plateau_seconds: float = 8.0      # ... and stops once it has gone this long without a new best
    max_seconds: float = 25.0         # hard cap per chain
    n_replicas: int = 12              # temperatures in the ladder
    t_low: float = 40.0               # coldest temperature (in score points)
    t_high: float = 8000.0            # hottest temperature
    steps_per_sweep: int = 400        # Metropolis steps per replica between swap attempts
    chunk_sweeps: int = 25            # sweeps per kernel call (the clock is checked between calls)
    order_penalty: float = 25000.0    # score penalty per bed-order violation inside a chain
    p_swap_move: int = 20             # percent of proposals that move/swap commands instead of re-rolling them
    chain_polish_depth: int = 2       # exhaustive k-opt polish applied to every chain result
    final_polish_depth: int = 3       # ... and to the best few candidates at the end
    final_polish_top: int = 3
    descent_nodes: int = 1_500_000    # node cap for each exact cat re-optimisation
    n_jobs: int = -1                  # joblib workers (-1 = all cores)
    seed: int = 12

    @staticmethod
    def preset(name: str) -> "SolveConfig":
        """``fast`` ~1 min, ``careful`` ~2 min, ``thorough`` ~6 min on a 2-core machine (timings are indicative)."""
        if name == "fast":
            return SolveConfig(chains_per_order=1, min_seconds=3, plateau_seconds=3, max_seconds=15,
                               chain_polish_depth=2, final_polish_depth=2, final_polish_top=2)
        if name == "careful":
            return SolveConfig()
        if name == "thorough":
            return SolveConfig(chains_per_order=6, min_seconds=8, plateau_seconds=8, max_seconds=25)
        raise ValueError(f"unknown preset {name!r} (use fast / careful / thorough)")


ALL_ORDERS = tuple(itertools.permutations(range(3)))


@dataclass
class ChainResult:
    order: Tuple[int, ...]
    seed: int
    plan: np.ndarray
    score: int
    seconds: float
    sweeps: int


@dataclass
class SolveResult:
    board: Board
    plan: Plan
    score: int
    cat_scores: List[int]
    arrival_order: List[int]
    arrival_turn: List[Optional[int]]
    seconds: float
    chains: List[ChainResult] = field(default_factory=list)

    def summary(self) -> str:
        names = [R.CAT_NAMES[i] for i in self.arrival_order]
        cats = ", ".join(f"{R.CAT_NAMES[i]} {s:,}" for i, s in enumerate(self.cat_scores))
        return (f"{self.board.name}: {self.score:,} points at ${self.plan.cost}  ({cats})\n"
                f"  bed order: {' -> '.join(names)}   finished in {self.seconds:.0f}s over {len(self.chains)} chains")


# =============================================================================================
# One parallel-tempering chain
# =============================================================================================
def _mix_seed(*parts: int) -> int:
    """splitmix64-style hash so nearby (seed, order, chain) triples give unrelated RNG streams."""
    x = 0x9E3779B97F4A7C15
    for p in parts:
        x = (x + (int(p) & 0xFFFFFFFFFFFFFFFF) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 31
    return x or 1


def run_chain(board: Board, rules: R.Rules, order: Optional[Sequence[int]], seed: int, cfg: SolveConfig,
              init_plan: Optional[np.ndarray] = None, solo: int = -1, budget: Optional[int] = None,
              t_high: Optional[float] = None, min_s: Optional[float] = None, plateau_s: Optional[float] = None,
              max_s: Optional[float] = None) -> ChainResult:
    """Run ONE parallel-tempering chain until it plateaus, then polish its best plan.

    order      : bed-arrival order to condition on (cat indices, first to last) or None
    init_plan  : start every replica from this plan instead of the empty plan ("seeded" chain)
    solo       : >= 0 runs the single-cat relaxation for that cat (see ``sim_kernel``)
    budget     : override the money limit (used by the single-cat searches)
    """
    t0 = time.time()
    min_s = cfg.min_seconds if min_s is None else min_s
    plateau_s = cfg.plateau_seconds if plateau_s is None else plateau_s
    max_s = cfg.max_seconds if max_s is None else max_s
    t_hi = cfg.t_high if t_high is None else t_high
    pack = board.numba_pack()
    flags = np.array(rules.as_flags(), dtype=np.int64)
    flags[5] = solo
    if budget is not None:
        flags[4] = budget
    n_slots, n_rep = board.n_slots, cfg.n_replicas
    scratch = make_scratch()

    # replica state
    plans = np.zeros((n_rep, n_slots), dtype=np.int8)
    if init_plan is not None:
        plans[:] = init_plan
    used = np.zeros((n_rep, n_slots), dtype=np.uint8)
    scores = np.zeros(n_rep, dtype=np.int64)
    costs = np.zeros(n_rep, dtype=np.int64)
    req = np.array(order if order is not None else (-1, -1, -1), dtype=np.int64)
    scratch_order, scratch_power = np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64)
    fits = np.zeros(n_rep)
    best_plan = np.zeros(n_slots, dtype=np.int8)
    best_info = np.array([-1, 0, 0], dtype=np.int64)      # best score, its cost, sweep it was found
    for k in range(n_rep):
        sc, co = sim_kernel(plans[k], pack, flags, used[k], scratch_order, scratch_power, scratch)
        for j in range(n_slots):                           # canonicalise the starting plan
            if not used[k, j]:
                plans[k, j] = 0
        scores[k], costs[k] = sc, co
        viol = 0 if req[0] < 0 else sum(1 for j in range(3) if scratch_order[j] >= 0 and scratch_order[j] != req[j])
        fits[k] = sc - cfg.order_penalty * viol
        if co <= flags[4] and sc > best_info[0]:
            best_info[:2] = sc, co
            best_plan[:] = plans[k]
    temps = cfg.t_low * (t_hi / cfg.t_low) ** (np.arange(n_rep) / (n_rep - 1))
    rng = np.array([_mix_seed(seed, solo + 1, *(order or (9, 9, 9)))], dtype=np.uint64)
    sweep_counter = np.zeros(1, dtype=np.int64)

    last_best, last_improve = -1, time.time()
    while True:
        pt_chunk(plans, used, scores, costs, fits, temps, best_plan, best_info, rng, req, pack, flags,
                 cfg.chunk_sweeps, cfg.steps_per_sweep, cfg.order_penalty, cfg.p_swap_move, sweep_counter, scratch)
        now = time.time()
        if best_info[0] > last_best:
            last_best, last_improve = int(best_info[0]), now
        elapsed = now - t0
        if elapsed >= max_s or (elapsed >= min_s and now - last_improve >= plateau_s):
            break

    polished, score, _ = polish_kernel(best_plan, pack, flags, cfg.chain_polish_depth, scratch)
    return ChainResult(tuple(order) if order is not None else (), seed, polished, int(score),
                       time.time() - t0, int(sweep_counter[0]))


def warm_up(board: Board, rules: R.Rules = R.Rules()) -> None:
    """Compile (or load from the on-disk cache) the numba kernels so worker processes do not all compile."""
    cfg = replace(SolveConfig(), min_seconds=0, plateau_seconds=0, max_seconds=0, chunk_sweeps=1,
                  steps_per_sweep=2, chain_polish_depth=3)
    run_chain(board, rules, (0, 1, 2), 1, cfg)


# =============================================================================================
# Public entry points
# =============================================================================================
def verify(board: Board, plan: Plan, rules: R.Rules = R.Rules()) -> int:
    """Re-play `plan` in BOTH simulators; raise if they disagree or the plan breaks the budget."""
    ref = simulate(board, plan, rules)
    fast_score, fast_spent, _, _, _ = FastSim(board, rules).evaluate(plan.cmds)
    if ref.score != fast_score or ref.cost != fast_spent:
        raise AssertionError(f"simulators disagree: reference {ref.score}/${ref.cost}, fast {fast_score}/${fast_spent}")
    if plan.cost > rules.budget:
        raise AssertionError(f"plan costs ${plan.cost}, over the ${rules.budget} budget")
    if plan.cost != ref.cost:
        raise AssertionError(f"plan places ${plan.cost} of commands but only ${ref.cost} ever fire (wasted commands)")
    return ref.score


def solve(board: Board, config: Optional[SolveConfig] = None, rules: R.Rules = R.Rules(),
          progress: bool = False) -> SolveResult:
    """Find a high-scoring plan for `board`.

    Pipeline: for each of the six bed-arrival orders run `chains_per_order` conditioned tempering chains in
    parallel -> polish each result -> take the best few -> deep polish -> verify in both simulators.
    """
    cfg = config or SolveConfig.preset("careful")
    problems = board.problems()
    if problems:
        raise ValueError(f"{board.name} is not a playable board: {'; '.join(problems)}")
    t0 = time.time()
    warm_up(board, rules)

    pool = Parallel(n_jobs=cfg.n_jobs, return_as="generator_unordered")

    def run_stage(label, tasks):
        stream = pool(tasks)
        if progress:
            from tqdm import tqdm
            stream = tqdm(stream, total=len(tasks), desc=f"{board.name}: {label}", unit="run")
        return list(stream)

    chains: List[ChainResult] = []

    # Stage 1 - chains conditioned on every arrival order (finds structures the single-cat view misses)
    orders = cfg.orders if cfg.orders is not None else ALL_ORDERS
    jobs = [(o, _mix_seed(cfg.seed, ci)) for o in orders for ci in range(cfg.chains_per_order)]
    chains += run_stage("order chains", [delayed(run_chain)(board, rules, o, sd, cfg) for o, sd in jobs])
    chains.sort(key=lambda ch: -ch.score)

    # deep polish + exact cat-neighbourhood descent of the few best distinct candidates
    pack, flags = board.numba_pack(), np.array(rules.as_flags(), dtype=np.int64)
    best_plan, best_score, seen = None, -1, set()
    for ch in chains:
        key = ch.plan.tobytes()
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > cfg.final_polish_top:
            break
        cand, score, _ = polish_kernel(ch.plan, pack, flags, cfg.final_polish_depth, make_scratch())
        cand, score = descend(board, cand, rules, cfg.descent_nodes)
        if score > best_score:
            best_plan, best_score = cand, int(score)

    plan = Plan(board, best_plan)
    verify(board, plan, rules)                      # raises if anything is inconsistent
    res = simulate(board, plan, rules)
    return SolveResult(board, plan, res.score, res.cat_scores, res.arrival_order, res.arrival_turn,
                       time.time() - t0, chains)


def solve_for(board: Board, minutes: float, config: Optional[SolveConfig] = None, rules: R.Rules = R.Rules(),
              progress: bool = False, on_improve=None) -> SolveResult:
    """Keep solving with fresh seeds until `minutes` have passed; return the best result.

    Chain success is a lottery (the best basin is hit by only a few percent of chains), so several independent
    solves beat one long one.  `on_improve(result)` is called whenever a better plan appears.
    """
    cfg = config or SolveConfig.preset("careful")
    deadline = time.time() + 60.0 * minutes
    best: Optional[SolveResult] = None
    k = 0
    while best is None or time.time() < deadline:
        res = solve(board, replace(cfg, seed=cfg.seed + 1000 * k), rules, progress)
        k += 1
        if best is None or res.score > best.score:
            best = res
            if on_improve:
                on_improve(res)
        if time.time() >= deadline:
            break
    return best
