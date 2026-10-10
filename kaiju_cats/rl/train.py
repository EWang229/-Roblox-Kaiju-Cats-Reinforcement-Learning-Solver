"""
Training and rollout code for the policy: REINFORCE with a *group baseline* (the "POMO" idea).

The algorithm in plain words
----------------------------
1. Draw a few fresh random boards.
2. Play every board many times (``rollouts`` games each) with the current policy, sampling commands at random
   according to the policy's probabilities.  Every game ends with a score.
3. For each board, compare each game with the *average of that board's games*.  Games that beat their own
   board's average get their commands made more likely; games below the average get them made less likely.
4. Repeat with new boards.

Why this algorithm (and not PPO, Q-learning, ...)
--------------------------------------------------
* The game is deterministic and the reward arrives only at the end.  A learned critic (PPO, A2C) has to
  predict the final score from a half-played game, which is hard and noisy.  Comparing a game with sibling
  games *on the same board* gives a baseline for free that is exactly right for that board: boards differ a lot
  in how much they can score (86k-312k), and the group mean removes that completely.
* Dividing by the group's spread makes every board count equally, whatever its score scale.
* Rollouts are cheap (the simulator is compiled), so we can afford big groups; big groups make the baseline
  accurate and also give the best-of-group score, which is what we use at solve time.
* Boards are generated fresh every iteration, so the network cannot memorise a map; it has to learn the game.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..board import Board
from ..generate import random_board
from .batch import MAX_SLOTS, BatchEnv
from .policy import PolicyNet

torch.set_num_threads(2)


# =============================================================================================
# Rolling out a policy
# =============================================================================================
@dataclass
class Rollouts:
    """Everything produced by playing a batch of games."""
    scores: np.ndarray                    # (n_games,) final score of every game
    plans: np.ndarray                     # (n_games, MAX_SLOTS) command chosen for every slot that fired
    spent: np.ndarray                     # (n_games,)
    # training data, only filled when record=True; one row per decision of a live game
    img: Optional[torch.Tensor] = None
    vec: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    act: Optional[torch.Tensor] = None
    game: Optional[torch.Tensor] = None   # which game each decision belongs to


@torch.no_grad()
def play(net: PolicyNet, env: BatchEnv, temperature: float = 1.0, greedy: bool = False, record: bool = False,
         generator: Optional[torch.Generator] = None) -> Rollouts:
    """Play every game of `env` to the end with `net` choosing the commands.

    temperature : >1 explores more, <1 sticks closer to the most likely command.
    greedy      : always take the most likely command (one deterministic game per board).
    record      : keep the observations/actions so a training step can be computed afterwards.
    """
    net.eval()
    env.reset()
    store: List[Tuple[torch.Tensor, ...]] = []
    while env.any_alive():
        img, vec, mask = env.observe()
        live = np.nonzero(env.alive)[0]
        t_img = torch.from_numpy(img[live])
        t_vec = torch.from_numpy(vec[live])
        t_mask = torch.from_numpy(mask[live])
        logp = net(t_img, t_vec, t_mask)
        if greedy:
            act = logp.argmax(dim=1)
        else:
            if temperature != 1.0:
                logp = torch.log_softmax(logp / temperature, dim=1)
            act = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(1)
        if record:
            store.append((t_img.clone(), t_vec.clone(), t_mask.clone(), act.clone(), torch.from_numpy(live)))
        full = np.zeros(env.n_games, dtype=np.int64)
        full[live] = act.numpy()
        env.step(full)
    out = Rollouts(env.scores.copy(), env.plans.copy(), env.spent())
    if record:
        out.img = torch.cat([s[0] for s in store])
        out.vec = torch.cat([s[1] for s in store])
        out.mask = torch.cat([s[2] for s in store])
        out.act = torch.cat([s[3] for s in store])
        out.game = torch.cat([s[4] for s in store])
    return out


# =============================================================================================
# The learning step
# =============================================================================================
def group_advantages(scores: np.ndarray, rollouts: int, floor: float = 0.03) -> np.ndarray:
    """Advantage of every game = (score - its board's mean) / (its board's spread).

    `floor` keeps boards where every game scored almost the same (spread ~ 0) from producing giant advantages.
    """
    s = scores.reshape(-1, rollouts).astype(np.float64)
    mean = s.mean(axis=1, keepdims=True)
    std = s.std(axis=1, keepdims=True)
    adv = (s - mean) / (np.maximum(std, floor * np.maximum(mean, 1.0)) + 1e-9)
    return adv.reshape(-1).astype(np.float32)


def policy_gradient_step(net: PolicyNet, opt: torch.optim.Optimizer, ro: Rollouts, adv: np.ndarray,
                         entropy_coef: float, chunk: int = 8192, clip_grad: float = 1.0) -> dict:
    """One REINFORCE update:  maximise  E[ advantage * sum_t log pi(a_t | s_t) ] + entropy_coef * entropy."""
    net.train()
    opt.zero_grad()
    n_games = len(adv)
    t_adv = torch.from_numpy(adv)
    m = ro.act.shape[0]
    ent_total, pg_total = 0.0, 0.0
    for lo in range(0, m, chunk):
        sl = slice(lo, min(m, lo + chunk))
        logp = net(ro.img[sl], ro.vec[sl], ro.mask[sl])
        chosen = logp.gather(1, ro.act[sl].unsqueeze(1)).squeeze(1)
        probs = logp.exp()
        ent = -(probs * logp.masked_fill(~ro.mask[sl], 0.0)).sum(dim=1)
        a = t_adv[ro.game[sl]]
        # every decision of a game shares that game's advantage; dividing by n_games averages over games
        pg = -(a * chosen).sum() / n_games
        # entropy is summed over a game's decisions and averaged over games, the same scaling as the pg term,
        # so `entropy_coef` is directly comparable with an advantage of size ~1
        loss = pg - entropy_coef * ent.sum() / n_games
        loss.backward()
        pg_total += float(pg.detach())
        ent_total += float(ent.sum().detach())
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
    opt.step()
    return {"pg": pg_total, "entropy": ent_total / m, "grad_norm": float(gn), "decisions": m}


# =============================================================================================
# Training loop
# =============================================================================================
@dataclass
class TrainConfig:
    iterations: int = 2000
    boards_per_iter: int = 16        # fresh boards per iteration
    rollouts: int = 32               # games per board  -> 512 games per iteration
    lr: float = 3e-4
    lr_final: float = 5e-5
    entropy_start: float = 0.05
    entropy_end: float = 0.002
    seed: int = 0
    log_every: int = 25
    save_path: Optional[str] = None
    save_every: int = 100
    width: int = 32
    blocks: int = 1


def fresh_boards(rng: random.Random, n: int) -> List[Board]:
    return [random_board(rng, name=f"train{rng.randrange(10**9)}") for _ in range(n)]


def train(cfg: TrainConfig, net: Optional[PolicyNet] = None, val_boards: Optional[Sequence[Board]] = None,
          val_every: int = 100, val_rollouts: int = 128, log=print) -> PolicyNet:
    """Train from scratch (or continue from `net`) on freshly generated random boards."""
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    net = net or PolicyNet(cfg.width, cfg.blocks)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    t0 = time.time()
    hist = []
    for it in range(1, cfg.iterations + 1):
        frac = it / cfg.iterations
        lr = cfg.lr_final + 0.5 * (cfg.lr - cfg.lr_final) * (1 + np.cos(np.pi * frac))     # cosine decay
        for g in opt.param_groups:
            g["lr"] = lr
        ent_coef = cfg.entropy_start + (cfg.entropy_end - cfg.entropy_start) * frac
        env = BatchEnv(fresh_boards(rng, cfg.boards_per_iter), rollouts=cfg.rollouts)
        ro = play(net, env, record=True)
        adv = group_advantages(ro.scores, cfg.rollouts)
        info = policy_gradient_step(net, opt, ro, adv, ent_coef)
        s = ro.scores.reshape(-1, cfg.rollouts)
        hist.append((s.mean(), s.max(axis=1).mean()))
        if it % cfg.log_every == 0:
            h = np.array(hist[-cfg.log_every:])
            log(f"it {it:5d}  mean {h[:, 0].mean():9.0f}  best-of-{cfg.rollouts} {h[:, 1].mean():9.0f}  "
                f"entropy {info['entropy']:.3f}  lr {lr:.1e}  {time.time() - t0:6.0f}s")
        if val_boards is not None and it % val_every == 0:
            res = evaluate(net, val_boards, rollouts=val_rollouts)
            log("   validation best-of-%d: %s" % (val_rollouts, " ".join(f"{x:,}" for x in res["best"])))
        if cfg.save_path and (it % cfg.save_every == 0 or it == cfg.iterations):
            save(net, cfg.save_path)
    return net


# =============================================================================================
# Evaluation and persistence
# =============================================================================================
def evaluate(net: PolicyNet, boards: Sequence[Board], rollouts: int = 128, temperature: float = 1.0) -> dict:
    """Best-of-`rollouts` sampled score on every board (plus the greedy game and the mean)."""
    env = BatchEnv(boards, rollouts=rollouts)
    ro = play(net, env, temperature=temperature)
    s = ro.scores.reshape(len(boards), rollouts)
    genv = BatchEnv(boards, rollouts=1)
    gro = play(net, genv, greedy=True)
    return {"best": [int(x) for x in s.max(axis=1)], "mean": [int(x) for x in s.mean(axis=1)],
            "greedy": [int(x) for x in gro.scores]}


def save(net: PolicyNet, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "width": net.width, "blocks": len(net.blocks)}, path)


def load(path) -> PolicyNet:
    ck = torch.load(path, map_location="cpu")
    net = PolicyNet(ck["width"], ck["blocks"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net
