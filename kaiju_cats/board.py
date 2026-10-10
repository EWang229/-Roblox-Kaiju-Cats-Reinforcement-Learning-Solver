"""
Board parsing and the "slot" model.

Text format (one row per line, whitespace separated tokens, ``#`` starts a comment line):

    ..   grass                     GB   gray building (2 floors x 250)
    YB   yellow building (2 x 500) PP   power plant (1 floor, doubles power)
    MD   mud                       SP   spikes
    RK   boulder                   --   water / nothing
    CB CR CG   start tile of the Blue / Red / Green cat   (grass underneath)
    bB bR bG   bed of the Blue / Red / Green cat

Example (the very first practice board):

    CB GB .. YB PP YB --
    -- .. GB YB MD GB bR
    CR GB .. SP PP GB bG
    -- .. RK GB GB GB bB
    CG GB .. YB PP YB --

A *slot* is a place a command can be attached: floor 0 (top, destroyed first) and floor 1 (bottom) of every
building, plus the single floor of every plant.  A plan is simply one command per slot (0 = nothing).
"""
from __future__ import annotations

import collections
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np

from . import rules as R

_TOKEN_TO_TILE = {
    "..": R.T_GRASS, "GB": R.T_LOW, "YB": R.T_HIGH, "PP": R.T_PLANT,
    "MD": R.T_MUD, "SP": R.T_SPIKE, "RK": R.T_BOULDER, "--": R.T_NONE,
}


@dataclass(frozen=True)
class Slot:
    """One attachment point for a command."""
    index: int
    row: int
    col: int
    floor: int        # 0 = top floor (destroyed first), 1 = bottom floor; plants only have floor 0
    tile: int         # T_LOW / T_HIGH / T_PLANT

    @property
    def label(self) -> str:
        what = {R.T_LOW: "gray", R.T_HIGH: "yellow", R.T_PLANT: "plant"}[self.tile]
        where = "" if self.tile == R.T_PLANT else (" top" if self.floor == 0 else " bottom")
        return f"row {self.row + 1}, col {self.col + 1} ({what}{where})"


class Board:
    """An immutable 5x7 Kaiju Cats map plus all the lookup tables the simulators need."""

    def __init__(self, tokens: List[List[str]], name: str = "board"):
        if len(tokens) != R.ROWS or any(len(row) != R.COLS for row in tokens):
            raise ValueError(f"a board must be {R.ROWS} rows x {R.COLS} columns")
        self.name = name
        self.tokens: Tuple[Tuple[str, ...], ...] = tuple(tuple(r) for r in tokens)

        self.tiles = np.full((R.ROWS, R.COLS), R.T_GRASS, dtype=np.int8)
        self.bed_owner = np.full((R.ROWS, R.COLS), -1, dtype=np.int8)
        self.starts = [None, None, None]
        self.beds = [None, None, None]
        for r, row in enumerate(tokens):
            for c, tk in enumerate(row):
                if tk in _TOKEN_TO_TILE:
                    self.tiles[r, c] = _TOKEN_TO_TILE[tk]
                elif len(tk) == 2 and tk[0] == "C" and tk[1] in R.CAT_LETTERS:
                    self.starts[R.CAT_LETTERS.index(tk[1])] = (r, c)      # a start tile is plain grass
                elif len(tk) == 2 and tk[0] == "b" and tk[1] in R.CAT_LETTERS:
                    cat = R.CAT_LETTERS.index(tk[1])
                    self.beds[cat] = (r, c)
                    self.tiles[r, c] = R.T_BED
                    self.bed_owner[r, c] = cat
                else:
                    raise ValueError(f"unknown token {tk!r} at row {r + 1}, col {c + 1}")
        if None in self.starts or None in self.beds:
            raise ValueError("every cat needs exactly one start tile (CB/CR/CG) and one bed (bB/bR/bG)")

        # Tie-break priority: when two cats fight with equal power the TOPMOST bed wins.  We store
        # "bigger number = wins ties" so the code never has to think about row direction again.
        self.prio = np.array([10 - self.beds[i][0] for i in range(3)], dtype=np.int64)

        # Slots, in row-major order.
        self.slots: List[Slot] = []
        self.slot_base = np.full((R.ROWS, R.COLS), -1, dtype=np.int64)  # first slot index of each tile
        for r in range(R.ROWS):
            for c in range(R.COLS):
                t = int(self.tiles[r, c])
                n_floors = 2 if t in (R.T_LOW, R.T_HIGH) else 1 if t == R.T_PLANT else 0
                if n_floors:
                    self.slot_base[r, c] = len(self.slots)
                    for f in range(n_floors):
                        self.slots.append(Slot(len(self.slots), r, c, f, t))
        self.n_slots = len(self.slots)

    # ---------------------------------------------------------------- constructors
    @classmethod
    def from_text(cls, text: str, name: str = "board") -> "Board":
        rows = [line.split() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        return cls(rows, name=name)

    @classmethod
    def from_file(cls, path) -> "Board":
        path = Path(path)
        return cls.from_text(path.read_text(), name=path.stem)

    # ---------------------------------------------------------------- views
    def to_text(self) -> str:
        return "\n".join(" ".join(row) for row in self.tokens)

    def signature(self) -> str:
        """Stable hash of the board layout; used as the key of the solution library."""
        return hashlib.sha1(self.to_text().encode()).hexdigest()[:12]

    def numba_pack(self):
        """The arrays handed to the numba kernels (tuple of plain ndarrays)."""
        sr = np.array([s[0] for s in self.starts], dtype=np.int64)
        sc = np.array([s[1] for s in self.starts], dtype=np.int64)
        slot_r = np.array([s.row for s in self.slots], dtype=np.int64)
        slot_c = np.array([s.col for s in self.slots], dtype=np.int64)
        return (self.tiles.astype(np.int64), self.bed_owner.astype(np.int64), sr, sc, self.prio,
                self.slot_base, slot_r, slot_c)

    # ---------------------------------------------------------------- sanity checks
    def problems(self) -> List[str]:
        """Return a list of reasons this board could not appear in the real game (empty list = playable)."""
        out = []
        for cat in range(3):
            r, c = self.starts[cat]
            if c + 1 < R.COLS and self.tiles[r, c + 1] == R.T_BOULDER:
                out.append(f"{R.CAT_NAMES[cat]} starts facing a boulder")
            if not self._can_reach_bed(cat):
                out.append(f"{R.CAT_NAMES[cat]} can never reach its bed (trapped)")
        return out

    def _can_reach_bed(self, cat: int) -> bool:
        """Relaxed reachability: a cat may be turned whenever it destroys a building/plant floor.

        State = (tile, facing direction).  Used only to reject impossible boards, never for scoring.
        """
        start, bed = self.starts[cat], self.beds[cat]

        def blocked(r, c):
            if not (0 <= r < R.ROWS and 0 <= c < R.COLS):
                return True
            t = int(self.tiles[r, c])
            if t in (R.T_NONE, R.T_BOULDER):
                return True
            return t == R.T_BED and int(self.bed_owner[r, c]) != cat

        seen = {(start, R.EAST)}
        queue = collections.deque(seen)
        while queue:
            (r, c), d = queue.popleft()
            nr, nc = r + R.DR[d], c + R.DC[d]
            if blocked(nr, nc):
                nxt = [((r, c), (d + 2) % 4)]
            elif (nr, nc) == bed:
                return True
            elif int(self.tiles[nr, nc]) in (R.T_LOW, R.T_HIGH, R.T_PLANT):
                nxt = [((nr, nc), nd) for nd in range(4)]
            else:
                nxt = [((nr, nc), d)]
            for s in nxt:
                if s not in seen:
                    seen.add(s)
                    queue.append(s)
        return False

    def __repr__(self):
        return f"Board({self.name!r}, slots={self.n_slots})"
