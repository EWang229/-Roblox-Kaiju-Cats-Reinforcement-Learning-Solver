"""
Solving a given board with the learned policy ("a board is sent to you").

Three stages, each reported separately so you can see what each one buys:

1. **Sample.**  Play thousands of games with the pretrained policy and keep the best one.  This costs a few
   seconds and uses nothing but what was learned on *other* boards.
2. **Adapt (active search).**  Keep training the policy, but only on this board: play a big group of games,
   push the policy toward the ones that beat the group average, repeat.  The network is a good starting
   point, so this converges far faster than learning from scratch.  It is still ordinary reinforcement
   learning, just with a single "environment".
3. **Polish.**  Take the best plan found and let the exact search (``lazy_dfs.descend``) re-route each cat
   perfectly while the others stay fixed.  RL finds the region; exact search squeezes out the last points.

Every plan is verified in the independent, readable simulator before it is returned.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from .. import rules as R
from ..board import Board
from ..lazy_dfs import canonical, descend
from ..plan import Plan
from ..simulator import simulate
from .batch import BatchEnv
from .policy import PolicyNet
from .exit import imitation_step, replay, run_expert_multi
from .train import play


@dataclass
class RLSolveResult:
    board: Board
    plan: Plan
    score: int
    sample_score: int                 # best game from the pretrained policy alone
    adapt_score: int                  # best after adapting to this board
    polish_score: int                 # after exact polishing (== score)
    seconds: float
    trace: List[tuple] = field(default_factory=list)      # (seconds, best score so far) during adaptation

    def summary(self) -> str:
        return (f"{self.board.name}: {self.score:,}  (policy sampling {self.sample_score:,} -> "
                f"adapted {self.adapt_score:,} -> polished {self.polish_score:,}) in {self.seconds:.0f}s")


DEFAULT_TEMPS = (0.8, 1.0, 1.3, 1.7, 2.2)


def rl_solve(board: Board, net: PolicyNet, seconds: float = 120.0, sample_games: int = 4096,
             lr: float = 3e-4, polish: bool = True, rules: R.Rules = R.Rules(), seed: int = 0, log=None,
             temps: Sequence[float] = DEFAULT_TEMPS, explore_share: float = 0.5, expert_top: int = 24,
             expert_polish: int = 8) -> RLSolveResult:
    """Solve `board` with the pretrained policy in about `seconds` seconds.

    Phase A - explore (first `explore_share` of the time): the policy is frozen.  Play thousands of games at a
        cycle of temperatures, let the expert (local search + exact re-routing) improve the best distinct ones.
        Different temperatures matter: a cold policy repeats its favourite plan, a hot one tries other
        structures; on one test board this alone lifted the result from 109k to 168k.
    Phase B - adapt (rest of the time, minus polishing): a copy of the policy is trained on this board by expert
        iteration - it imitates the best few plans found so far (the elite set, which phase A filled) and then
        samples again, so the search concentrates around the strongest structures.
    Phase C - polish: exact re-routing of the single best plan.
    """
    t0 = time.time()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    best_score, best_cmds = -1, np.zeros(board.n_slots, dtype=np.int8)
    trace: List[tuple] = []

    def note(score: int, plan: np.ndarray, tag: str) -> None:
        nonlocal best_score, best_cmds
        if score > best_score:
            best_score, best_cmds = int(score), plan.copy()
            trace.append((time.time() - t0, best_score))
            if log:
                log(f"  {time.time() - t0:6.1f}s  best {best_score:,} ({tag})")

    # ---- stage 0: plain sampling from the pretrained policy (reported separately) -------------------------
    ro = play(net, BatchEnv([board], rollouts=sample_games, rules=rules), generator=gen)
    note(int(ro.scores.max()), ro.plans[int(ro.scores.argmax()), :board.n_slots], "policy sample")
    sample_score = best_score

    work = copy.deepcopy(net)
    opt = torch.optim.Adam(work.parameters(), lr=lr)
    env_explore = BatchEnv([board], rollouts=4096, rules=rules)
    env_adapt = BatchEnv([board], rollouts=2048, rules=rules)
    elites: List[Tuple[int, np.ndarray]] = []             # (score, plan) of the best distinct plans found so far
    t_polish = 0.1 * seconds if polish else 0.0
    t_explore_end = explore_share * (seconds - t_polish)
    n_iter = 0
    adapt_started = None
    while time.time() - t0 < seconds - t_polish:
        exploring = time.time() - t0 < t_explore_end
        if not exploring and adapt_started is None:
            adapt_started = best_score
        temp = temps[n_iter % len(temps)] if exploring else 1.0
        n_iter += 1
        ro = play(work, env_explore if exploring else env_adapt, temperature=temp, generator=gen)
        i = int(ro.scores.argmax())
        note(int(ro.scores[i]), ro.plans[i, :board.n_slots], "sample")
        for plan, score in run_expert_multi(board, ro, top=expert_top, polish_top=expert_polish, rules=rules):
            note(score, plan, "after expert")
            if all(plan.tobytes() != e[1].tobytes() for e in elites):
                elites.append((int(score), plan.copy()))
        elites.sort(key=lambda e: -e[0])
        del elites[8:]
        if not exploring:                                   # phase B: teach the policy the elite plans
            targets = replay([board] * len(elites), [e[1] for e in elites], rules)
            for _ in range(4):
                imitation_step(work, opt, targets)
    adapt_score = best_score

    # ---- stage 3: exact polish --------------------------------------------------------------------
    if polish:
        best_cmds, polished = descend(board, best_cmds, rules)
        best_score = max(best_score, int(polished))
    plan = Plan(board, canonical(board, best_cmds, rules))     # drop commands that never fire (they cost budget)
    res = simulate(board, plan, rules)                          # independent check
    if res.score != best_score or plan.cost > rules.budget:
        raise AssertionError(f"verification failed: rollout {best_score}, simulator {res.score}, cost ${plan.cost}")
    return RLSolveResult(board, plan, res.score, sample_score, adapt_score, res.score, time.time() - t0, trace)
