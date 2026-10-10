"""Random board generator in the style of the real game (used for benchmarks and RL training).

Observed pattern of the real boards:
  * cats start in column 1 on rows 1/3/5, beds sit in column 7 on rows 2/3/4 - only the ORDER of the cats changes
  * the 5x5 interior holds buildings, plants, mud, spikes, a boulder and some grass
  * a board that would trap a cat (or put a boulder right in front of a start/bed) never appears
"""
from __future__ import annotations

import random
from typing import Optional

from . import rules as R
from .board import Board


def random_board(rng: Optional[random.Random] = None, name: str = "random") -> Board:
    rng = rng or random.Random()
    for _ in range(10_000):
        n_plant = rng.choice([2, 3, 3, 3, 4])
        n_mud = rng.choice([0, 1, 1, 1, 2])
        n_spike = rng.choice([0, 1, 1, 1, 2])
        n_rock = rng.choice([0, 1, 1, 1, 2])
        n_grass = rng.randint(3, 7)
        n_build = 25 - (n_plant + n_mud + n_spike + n_rock + n_grass)
        n_high = round(n_build * rng.uniform(0.25, 0.5))
        tiles = (["PP"] * n_plant + ["MD"] * n_mud + ["SP"] * n_spike + ["RK"] * n_rock + [".."] * n_grass +
                 ["YB"] * n_high + ["GB"] * (n_build - n_high))
        rng.shuffle(tiles)
        grid = [["--"] * R.COLS for _ in range(R.ROWS)]
        it = iter(tiles)
        for r in range(R.ROWS):
            for c in range(1, 6):
                grid[r][c] = next(it)
        starts, beds = list("BRG"), list("BRG")
        rng.shuffle(starts)
        rng.shuffle(beds)
        for row, cat in zip((0, 2, 4), starts):
            grid[row][0] = "C" + cat
        for row, cat in zip((1, 2, 3), beds):
            grid[row][6] = "b" + cat
        if any(grid[r][5] == "RK" for r in (1, 2, 3)):           # boulder blocking a bed entrance
            continue
        board = Board(grid, name=name)
        if not board.problems():
            return board
    raise RuntimeError("could not generate a playable board")
