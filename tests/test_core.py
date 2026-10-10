"""Core correctness tests.  The in-game replay is the ground truth the whole model was checked against."""
import random
from pathlib import Path

import numpy as np
import pytest

from kaiju_cats import Board, Plan, library, simulate
from kaiju_cats.fastsim import FastSim
from kaiju_cats.generate import random_board

ROOT = Path(__file__).resolve().parents[1]
BOARD2 = ROOT / "boards" / "board_02.txt"

REPLAY_PLAN = ROOT / "tests" / "data" / "replay_board2.plan"


def test_matches_the_real_game_replay():
    """This plan was played in the actual game: Blue 3,500 / Red 145,000 / Green 9,000 = 157,500."""
    b = Board.from_file(BOARD2)
    res = simulate(b, Plan.from_file(b, REPLAY_PLAN))
    assert res.cat_scores == [3_500, 145_000, 9_000]
    assert res.score == 157_500 and res.cost == 200


def test_board_parses_and_roundtrips():
    b = Board.from_file(BOARD2)
    assert b.problems() == []
    assert Board.from_text(b.to_text()).signature() == b.signature()


def test_library_plans_verify_in_both_simulators():
    entries = library.load()
    assert len(entries) >= 8
    for e in entries.values():
        b = Board.from_text(e["board"], name=e["name"])
        found = library.lookup(b)
        assert found is not None, f"{e['name']}: stored plan no longer reproduces its score"
        plan, entry = found
        assert simulate(b, plan).score == entry["score"]
        assert FastSim(b).evaluate(plan.cmds)[0] == entry["score"]
        assert plan.cost <= 200


def test_best_known_board2_score():
    b = Board.from_file(BOARD2)
    plan, entry = library.lookup(b)
    assert entry["score"] >= 193_375


@pytest.mark.parametrize("seed", range(5))
def test_fast_and_readable_simulators_agree_on_random_plans(seed):
    rng = random.Random(seed)
    b = random_board(rng, name=f"r{seed}")
    fs = FastSim(b)
    nrng = np.random.default_rng(seed)
    for _ in range(200):
        cmds = np.where(nrng.random(b.n_slots) < 0.5, nrng.integers(0, 7, b.n_slots), 0).astype(np.int8)
        res = simulate(b, Plan(b, cmds))
        score, spent, *_ = fs.evaluate(cmds)
        assert (score, spent) == (res.score, res.cost)


def test_plan_file_roundtrip(tmp_path):
    b = Board.from_file(BOARD2)
    plan, _ = library.lookup(b)
    f = tmp_path / "p.plan"
    plan.to_file(f)
    assert Plan.from_file(b, f) == plan


def test_exact_cat_reoptimisation_never_worsens_and_matches_simulator():
    from kaiju_cats.lazy_dfs import reoptimise, slot_owners
    b = Board.from_file(BOARD2)
    plan, entry = library.lookup(b)
    fs = FastSim(b)
    owner = np.full(b.n_slots, -1, dtype=np.int64)
    slot_owners(plan.cmds.astype(np.int8), b.numba_pack(), fs.flags, owner)
    for cat in range(3):
        score, cmds, nodes, finished = reoptimise(b, plan.cmds, (owner == cat).astype(np.uint8), node_cap=200_000)
        assert score >= entry["score"]
        assert fs.evaluate(cmds)[0] == score


def test_gymnasium_env_smoke():
    gym = pytest.importorskip("gymnasium")
    from kaiju_cats.env import KaijuCatsEnv
    env = KaijuCatsEnv(Board.from_file(BOARD2), seed=0)
    obs, info = env.reset(seed=0)
    done = False
    steps = 0
    while not done and steps < 200:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1
    assert done
