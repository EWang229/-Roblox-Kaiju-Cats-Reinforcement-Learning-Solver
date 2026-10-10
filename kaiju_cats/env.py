"""
Gymnasium environment for Kaiju Cats.

The game is deterministic and the whole plan can be searched directly (that is what :mod:`kaiju_cats.solver` does),
but if you want to try reinforcement learning the natural formulation is a **lazy-decision MDP**:

    A command only matters at the instant its floor is destroyed, so the agent is asked for a command
    exactly then - "cat Green just broke the top floor of a yellow building, what should it do?"

* observation : flat float32 vector (turn, money, who/where the event is, every cat, tile map, floors left)
* action      : Discrete(7) - none, N, E, S, W, Stomp, Power Up  (see ``action_masks`` for what is affordable)
* reward      : ``reward_mode="sparse"`` -> final score / 100k at the end of the episode
                ``reward_mode="dense"``  -> change in total power after each decision (sums to the same thing)
* episodes    : one board per episode; pass a list of boards or a callable to train for generalisation

Every episode is exactly "choose one command per destroyed floor, in the order floors get destroyed".
This is a much smaller search space than "place commands on all floors up front", but the solver's static-plan
search still wins on every board we tried (see docs/INSIGHTS.md), so treat this as a playground, not a shortcut.
"""
from __future__ import annotations

import random
from typing import Callable, List, Optional, Sequence, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import rules as R
from .board import Board
from .generate import random_board
from .simulator import Game

BoardSource = Union[Board, Sequence[Board], Callable[[random.Random], Board]]


class KaijuCatsEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, boards: Optional[BoardSource] = None, reward_mode: str = "sparse",
                 rules: R.Rules = R.Rules(), seed: Optional[int] = None):
        super().__init__()
        self._source = boards if boards is not None else random_board
        self.reward_mode = reward_mode
        self.rules = rules
        self._rng = random.Random(seed)
        self.action_space = spaces.Discrete(R.N_COMMANDS)
        self._obs_size = 2 + 3 + 3 + 1 + 3 * (2 + 4 + 5 + 1) + R.ROWS * R.COLS * 9 + R.ROWS * R.COLS
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self._obs_size,), dtype=np.float32)
        self.game: Optional[Game] = None
        self.board: Optional[Board] = None

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
        src = self._source
        self.board = src if isinstance(src, Board) else (self._rng.choice(list(src)) if isinstance(src, (list, tuple))
                                                         else src(self._rng))
        self.game = Game(self.board, self.rules)
        self.game.advance()
        self._last_power = 0
        if not self.game.pending:                     # nothing ever gets destroyed: trivial episode
            return self._obs(), {"board": self.board.name, "terminal_score": self.game.finish().score}
        return self._obs(), {"board": self.board.name}

    def action_masks(self) -> np.ndarray:
        """Boolean mask of affordable commands (for sb3-contrib MaskablePPO and friends)."""
        left = self.rules.budget - self.game.spent
        return np.array([R.CMD_COST[c] <= left for c in range(R.N_COMMANDS)], dtype=bool)

    def step(self, action: int):
        g = self.game
        cmd = int(action)
        invalid = R.CMD_COST[cmd] > self.rules.budget - g.spent
        if invalid:
            cmd = R.CMD_NONE                          # unaffordable -> silently "no command", flagged in info
        g.decide(cmd)
        if not g.pending:
            g.advance()
        done = not g.pending
        total = sum(0 if c.status == 4 else c.power for c in g.cats)
        if self.reward_mode == "dense":
            reward = (total - self._last_power) / 1e5
        else:
            reward = (total / 1e5) if done else 0.0
        self._last_power = total
        info = {"invalid_action": invalid}
        if done:
            info["score"] = total
        return self._obs(), float(reward), done, False, info

    # ------------------------------------------------------------------ observation
    def _obs(self) -> np.ndarray:
        g, b = self.game, self.board
        v = np.zeros(self._obs_size, dtype=np.float32)
        i = 0
        v[i] = g.turn / self.rules.turns; v[i + 1] = (self.rules.budget - g.spent) / self.rules.budget; i += 2
        ev = g.pending[0] if g.pending else None
        if ev is not None:
            v[i + ev.cat] = 1.0
            v[i + 3 + {R.T_LOW: 0, R.T_HIGH: 1, R.T_PLANT: 2}[ev.tile]] = 1.0
            v[i + 6 + 0] = g.floors[(ev.row, ev.col)] / 2.0       # floors LEFT after the one just destroyed
        i += 3 + 3 + 1
        for cat in g.cats:
            v[i] = cat.row / (R.ROWS - 1); v[i + 1] = cat.col / (R.COLS - 1)
            v[i + 2 + cat.direction] = 1.0
            v[i + 6 + cat.status] = 1.0
            v[i + 11] = min(1.0, np.log1p(cat.power) / np.log(1e6))
            i += 12
        for r in range(R.ROWS):
            for c in range(R.COLS):
                v[i + int(b.tiles[r, c])] = 1.0
                i += 9
        for r in range(R.ROWS):
            for c in range(R.COLS):
                v[i] = g.floors.get((r, c), 0) / 2.0
                i += 1
        return v

    def render(self):
        g = self.game
        cats = ", ".join(f"{R.CAT_NAMES[k]} {c.power}" for k, c in enumerate(g.cats))
        return f"{self.board.name} turn {g.turn} ${self.rules.budget - g.spent} left | {cats}"
