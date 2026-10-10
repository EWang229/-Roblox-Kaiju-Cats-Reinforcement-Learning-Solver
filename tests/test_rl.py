"""Tests for the reinforcement-learning pieces.  The environment tests need only numpy/numba; the rest need torch."""
import random
from pathlib import Path

import numpy as np
import pytest

from kaiju_cats import Board, Plan, library, simulate
from kaiju_cats.generate import random_board
from kaiju_cats.rl.baselines import random_search
from kaiju_cats.rl.batch import BatchEnv

ROOT = Path(__file__).resolve().parents[1]
BOARD2 = ROOT / "boards" / "board_02.txt"


def _random_rollouts(env, seed):
    rng = np.random.default_rng(seed)
    env.reset()
    while env.any_alive():
        _, _, mask = env.observe()
        p = mask.astype(float)
        p /= p.sum(axis=1, keepdims=True)
        env.step(np.array([rng.choice(len(r), p=r) for r in p]))


def test_batch_env_matches_the_simulator_on_random_games():
    boards = [Board.from_file(BOARD2)] + [random_board(random.Random(i)) for i in range(3)]
    env = BatchEnv(boards, rollouts=24)
    _random_rollouts(env, 0)
    for g in range(env.n_games):
        b = boards[env.bidx[g]]
        res = simulate(b, Plan(b, env.plans[g, :b.n_slots]))
        assert res.score == env.scores[g]
        assert res.cost == env.spent()[g] <= 200


def test_masks_never_offer_unaffordable_or_pointless_commands():
    env = BatchEnv([Board.from_file(BOARD2)], rollouts=32)
    rng = np.random.default_rng(1)
    env.reset()
    while env.any_alive():
        _, _, mask = env.observe()
        live = env.alive
        assert mask[live, 0].all()                                     # "no command" is always allowed
        left = 200 - env.spent()
        for a, cost in enumerate((0, 10, 10, 10, 10, 20, 30)):
            assert not (mask[live, a] & (cost > left[live])).any()
        p = mask.astype(float)
        p /= p.sum(axis=1, keepdims=True)
        env.step(np.array([rng.choice(7, p=r) for r in p]))


def test_replaying_a_library_plan_through_the_env_reproduces_its_score():
    """Feed the stored best plan's commands back as actions: the env must give exactly the stored score."""
    b = Board.from_file(BOARD2)
    plan, entry = library.lookup(b)
    env = BatchEnv([b], rollouts=1)
    env.reset()
    while env.any_alive():
        env.observe()
        slot = env.pending_slots()[0]
        env.step(np.array([int(plan.cmds[slot])]))
    assert env.scores[0] == entry["score"]


def test_random_search_baseline_is_valid():
    b = Board.from_file(BOARD2)
    score, cmds = random_search(b, 256)
    assert simulate(b, Plan(b, cmds)).score == score


def test_training_smoke_and_rl_solve(tmp_path):
    pytest.importorskip("torch")
    from kaiju_cats.rl.policy import PolicyNet
    from kaiju_cats.rl.solve import rl_solve
    from kaiju_cats.rl.train import TrainConfig, evaluate, load, save, train

    net = train(TrainConfig(iterations=3, boards_per_iter=2, rollouts=8, log_every=1), log=lambda *_: None)
    p = tmp_path / "p.pt"
    save(net, p)
    net2 = load(p)
    b = Board.from_file(BOARD2)
    assert evaluate(net, [b], rollouts=16)["greedy"] == evaluate(net2, [b], rollouts=16)["greedy"]
    res = rl_solve(b, net2, seconds=8, sample_games=64)
    assert res.score == simulate(b, res.plan).score and res.plan.cost <= 200


def test_expert_iteration_smoke():
    pytest.importorskip("torch")
    from kaiju_cats.rl.exit import ExItConfig, train_exit
    net = train_exit(ExItConfig(iterations=2, boards_per_iter=2, rollouts=8, exact_nodes=2000, updates=2, batch=256,
                                log_every=1), log=lambda *_: None)
    assert sum(p.numel() for p in net.parameters()) > 0


def test_shipped_policy_is_much_better_than_random_play():
    pytest.importorskip("torch")
    from kaiju_cats.rl.train import load
    from kaiju_cats.rl.train import play
    policy_path = ROOT / "kaiju_cats" / "data" / "policy.pt"
    if not policy_path.exists():
        pytest.skip("no shipped policy")
    b = Board.from_file(BOARD2)
    net = load(policy_path)
    learned = play(net, BatchEnv([b], rollouts=512)).scores.max()
    random_best, _ = random_search(b, 512)
    assert learned >= 2 * random_best
