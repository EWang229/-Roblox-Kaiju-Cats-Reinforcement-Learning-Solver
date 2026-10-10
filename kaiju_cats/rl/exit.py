"""
Expert iteration: reinforcement learning where *search improves the policy's own games*.

Why this exists
---------------
Plain policy gradient (``train.py``) learns slowly here.  The reward arrives only at the end of a game, and a
good game needs a long, exact chain of decisions (save money, put Power Ups on the right floors, steer one cat
to its bed on the very last turn).  Almost every random game is bad, so the learning signal is weak.

Expert iteration (Anthony et al. 2017, the idea behind AlphaZero) fixes the signal instead of the algorithm:

    1. Let the current policy play many games on a board.               (the "apprentice")
    2. Take its best few games and improve them with a cheap local search -
       an exhaustive "change any 1-2 commands" hill-climb.                (the "expert")
    3. Train the policy to imitate the improved games.                    (supervised learning)
    4. Repeat on new boards.

Each round the expert starts from what the apprentice can already do and pushes a little beyond it, and the
apprentice then absorbs that.  The apprentice never needs a lucky 1-in-10,000 game to learn from.

What the expert is allowed to use: only the game simulator (the same information the policy has) and a small
amount of compute.  No stored solutions and no knowledge of the 8 real boards.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from .. import rules as R
from ..board import Board
from concurrent.futures import ThreadPoolExecutor
import os

from ..fastsim import make_scratch, polish_kernel, polish_nogil
from ..lazy_dfs import descend
from .batch import BatchEnv
from .policy import PolicyNet
from .train import (Rollouts, evaluate, fresh_boards, group_advantages, play, policy_gradient_step, save)


def replay(boards: Sequence[Board], plans: Sequence[np.ndarray], rules: R.Rules = R.Rules()) -> Rollouts:
    """Play each plan through the lazy-decision environment, recording what the policy would have seen.

    This turns a finished plan ("put Power Up on floor 7") back into (observation, action) pairs, which is
    the form supervised learning needs.  Each plan is replayed on its own board; ``game`` indexes `boards`.
    """
    env = BatchEnv(boards, rollouts=1, rules=rules)
    env.reset()
    pad = np.zeros((len(boards), 64), dtype=np.int64)
    for i, p in enumerate(plans):
        pad[i, :len(p)] = p
    store = []
    while env.any_alive():
        img, vec, mask = env.observe()
        live = np.nonzero(env.alive)[0]
        slots = env.pending_slots()
        act = pad[live, slots[live]]
        # The expert's search can leave harmless "extra" commands (e.g. turning a cat toward the direction it
        # already faces).  The policy's mask forbids those, so they are replaced by "no command" - the game
        # outcome cannot get worse (it only saves money) and the target stays something the policy can output.
        ok = mask[live, act]
        act = np.where(ok, act, 0)
        store.append((torch.from_numpy(img[live]).clone(), torch.from_numpy(vec[live]).clone(),
                      torch.from_numpy(mask[live]).clone(), torch.from_numpy(act.copy()),
                      torch.from_numpy(live)))
        full = np.zeros(env.n_games, dtype=np.int64)
        full[live] = act
        env.step(full)
    out = Rollouts(env.scores.copy(), env.plans.copy(), env.spent())
    out.img = torch.cat([s[0] for s in store])
    out.vec = torch.cat([s[1] for s in store])
    out.mask = torch.cat([s[2] for s in store])
    out.act = torch.cat([s[3] for s in store])
    out.game = torch.cat([s[4] for s in store])
    return out


def expert_improve(board: Board, plan: np.ndarray, depth: int = 2, rules: R.Rules = R.Rules()) -> Tuple[np.ndarray, int]:
    """The expert: exhaustive k-change hill-climb on a finished plan (compiled, tens of milliseconds)."""
    pack = board.numba_pack()
    flags = np.array(rules.as_flags(), dtype=np.int64)
    improved, score, _ = polish_nogil(np.ascontiguousarray(plan, dtype=np.int8), pack, flags, depth, make_scratch())
    return improved, int(score)


_POOL = ThreadPoolExecutor(max_workers=max(1, os.cpu_count() or 1))


def run_expert(boards: Sequence[Board], ro: Rollouts, rollouts: int, top: int, depth: int,
               rules: R.Rules = R.Rules(), exact_nodes: int = 0) -> List[Tuple[np.ndarray, int]]:
    """For every board: cheap 1-change climb on its `top` best distinct games, then the strong `depth`-change
    climb on the winner.  Boards are processed in parallel threads (the compiled search releases the GIL).

    Why two levels?  The 1-change climb costs about a millisecond and tells us which of the apprentice's games
    has the most potential; the 2-change climb is ~1000x dearer but far stronger (it roughly triples a typical
    early-training score), so it is spent on one game per board.
    """
    s = ro.scores.reshape(len(boards), rollouts)

    def work(bi: int) -> Tuple[np.ndarray, int]:
        b = boards[bi]
        seen, cands = set(), []
        for j in np.argsort(-s[bi]):
            plan = ro.plans[bi * rollouts + j, :b.n_slots]
            key = plan.tobytes()
            if key in seen:
                continue
            seen.add(key)
            cands.append(expert_improve(b, plan, 1, rules))
            if len(seen) >= top:
                break
        best_plan, _ = max(cands, key=lambda x: x[1])
        plan2, score2 = expert_improve(b, best_plan, depth, rules)
        if exact_nodes > 0:      # then let the exact search re-route each cat perfectly (the strongest move we have)
            plan3, score3 = descend(b, plan2, rules, node_cap=exact_nodes, table_bits=18, nogil=True)
            if score3 > score2:
                return plan3, int(score3)
        return plan2, score2

    return list(_POOL.map(work, range(len(boards))))


def imitation_step(net: PolicyNet, opt: torch.optim.Optimizer, data: Rollouts, chunk: int = 8192,
                   clip_grad: float = 1.0) -> dict:
    """Supervised update: raise the probability of the expert's commands (cross-entropy)."""
    net.train()
    opt.zero_grad()
    m = data.act.shape[0]
    total = 0.0
    for lo in range(0, m, chunk):
        sl = slice(lo, min(m, lo + chunk))
        logp = net(data.img[sl], data.vec[sl], data.mask[sl])
        loss = -logp.gather(1, data.act[sl].unsqueeze(1)).sum() / m
        loss.backward()
        total += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
    opt.step()
    return {"ce": total}


@dataclass
class ExItConfig:
    iterations: int = 600
    boards_per_iter: int = 16
    rollouts: int = 64               # games the apprentice plays per board
    top: int = 3                     # best games per board handed to the expert
    buffer: int = 150_000            # decisions kept in the replay buffer (older expert games are dropped)
    updates: int = 8                 # gradient steps per iteration, each on a random minibatch of the buffer
    batch: int = 4096
    expert_depth: int = 2
    exact_nodes: int = 100_000       # node cap of the exact cat re-routing after the climb (0 = off)
    lr: float = 5e-4
    lr_final: float = 1e-4
    pg_weight: float = 0.0           # optional policy-gradient term on the apprentice's own games
    entropy: float = 0.02
    seed: int = 0
    log_every: int = 10
    val_every: int = 50
    save_path: Optional[str] = None
    width: int = 48
    blocks: int = 1


def train_exit(cfg: ExItConfig, net: Optional[PolicyNet] = None, val_boards: Optional[Sequence[Board]] = None,
               val_rollouts: int = 256, log=print) -> PolicyNet:
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    net = net or PolicyNet(cfg.width, cfg.blocks)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    t0 = time.time()
    hist: List[Tuple[float, float, float]] = []
    buf = Rollouts(None, None, None)
    for it in range(1, cfg.iterations + 1):
        frac = it / cfg.iterations
        lr = cfg.lr_final + 0.5 * (cfg.lr - cfg.lr_final) * (1 + np.cos(np.pi * frac))
        for g in opt.param_groups:
            g["lr"] = lr
        boards = fresh_boards(rng, cfg.boards_per_iter)
        env = BatchEnv(boards, rollouts=cfg.rollouts)
        ro = play(net, env, record=cfg.pg_weight > 0)
        s = ro.scores.reshape(len(boards), cfg.rollouts)
        # ---- the expert improves each board's best apprentice games ---------------------------------
        improved = run_expert(boards, ro, cfg.rollouts, cfg.top, cfg.expert_depth, exact_nodes=cfg.exact_nodes)
        tb, tp = list(boards), [p for p, _ in improved]
        expert_scores = [sc for _, sc in improved]
        data = replay(tb, tp)
        # Replay buffer: expert games are expensive (the expert is the slow part), so each one is learned from
        # several times, mixed with older ones.  This is ordinary "experience replay" applied to expert targets.
        for name in ("img", "vec", "mask", "act"):
            old = getattr(buf, name)
            new = getattr(data, name)
            setattr(buf, name, new if old is None else torch.cat([old, new])[-cfg.buffer:])
        n_buf = buf.act.shape[0]
        for _ in range(cfg.updates):
            idx = torch.randint(0, n_buf, (min(cfg.batch, n_buf),))
            mini = Rollouts(None, None, None, buf.img[idx], buf.vec[idx], buf.mask[idx], buf.act[idx])
            info = imitation_step(net, opt, mini)
        if cfg.pg_weight > 0:
            adv = group_advantages(ro.scores, cfg.rollouts) * cfg.pg_weight
            policy_gradient_step(net, opt, ro, adv, cfg.entropy)
        hist.append((s.mean(), s.max(axis=1).mean(), float(np.mean(expert_scores))))
        if it % cfg.log_every == 0:
            h = np.array(hist[-cfg.log_every:]).mean(axis=0)
            log(f"it {it:5d}  apprentice mean {h[0]:9.0f}  best-of-{cfg.rollouts} {h[1]:9.0f}  "
                f"after expert {h[2]:9.0f}  ce {info['ce']:.3f}  lr {lr:.1e}  {time.time() - t0:6.0f}s")
        if val_boards is not None and it % cfg.val_every == 0:
            res = evaluate(net, val_boards, rollouts=val_rollouts)
            log("   validation best-of-%d: %s" % (val_rollouts, " ".join(f"{x:,}" for x in res["best"])))
        if cfg.save_path and (it % 50 == 0 or it == cfg.iterations):
            save(net, cfg.save_path)
    return net


def run_expert_multi(board: Board, ro: Rollouts, top: int = 8, polish_top: int = 4, depth: int = 2,
                     exact_nodes: int = 100_000, rules: R.Rules = R.Rules()) -> List[Tuple[np.ndarray, int]]:
    """Single-board version used at solve time: improve several distinct games (not just the winner) in parallel.

    The best *sampled* game is not always the one with the best *improved* score, so spending the expert on a few
    different candidates finds better plans and gives the policy more varied targets.
    Returns improved (plan, score) pairs, best first.
    """
    seen, cands = set(), []
    for j in np.argsort(-ro.scores):
        plan = ro.plans[j, :board.n_slots]
        key = plan.tobytes()
        if key in seen:
            continue
        seen.add(key)
        cands.append(expert_improve(board, plan, 1, rules))
        if len(seen) >= top:
            break
    cands.sort(key=lambda x: -x[1])

    def work(item: Tuple[np.ndarray, int]) -> Tuple[np.ndarray, int]:
        plan, score = expert_improve(board, item[0], depth, rules)
        if exact_nodes > 0:
            p3, s3 = descend(board, plan, rules, node_cap=exact_nodes, table_bits=18, nogil=True)
            if s3 > score:
                return p3, int(s3)
        return plan, score

    out = list(_POOL.map(work, cands[:polish_top]))
    out.sort(key=lambda x: -x[1])
    return out
