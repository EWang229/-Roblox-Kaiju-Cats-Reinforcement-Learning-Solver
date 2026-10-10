"""A *plan* is one command per slot.  This module holds the Plan class and all the pretty-printing."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import rules as R
from .board import Board


class Plan:
    """Commands attached to a board's slots.

    ``cmds[i]`` is the command (0..6, see :mod:`kaiju_cats.rules`) attached to ``board.slots[i]``.
    """

    def __init__(self, board: Board, cmds=None):
        self.board = board
        self.cmds = np.zeros(board.n_slots, dtype=np.int8) if cmds is None else np.asarray(cmds, dtype=np.int8).copy()
        if self.cmds.shape != (board.n_slots,):
            raise ValueError("plan length does not match the board's slot count")

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_placements(cls, board: Board, placements: Dict[Tuple[int, int, int], int]) -> "Plan":
        """``placements[(row, col, floor)] = command`` using 0-indexed row/col and floor 0 = top."""
        plan = cls(board)
        for (r, c, f), cmd in placements.items():
            base = int(board.slot_base[r, c])
            if base < 0:
                raise ValueError(f"tile ({r + 1},{c + 1}) cannot hold commands")
            plan.cmds[base + f] = cmd
        return plan

    @classmethod
    def from_file(cls, board: Board, path) -> "Plan":
        """Read lines of ``row col floor cmd`` (0-indexed) - the format used by the original C++ solver."""
        placements = {}
        for line in Path(path).read_text().splitlines():
            parts = line.split()
            if len(parts) == 4:
                r, c, f, cmd = map(int, parts)
                placements[(r, c, f)] = cmd
        return cls.from_placements(board, placements)

    def to_file(self, path) -> None:
        lines = [f"{s.row} {s.col} {s.floor} {int(self.cmds[s.index])}" for s in self.board.slots if self.cmds[s.index]]
        Path(path).write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------------ summaries
    @property
    def cost(self) -> int:
        """Money spent on the commands that are placed (what the game charges you)."""
        return int(sum(R.CMD_COST[int(c)] for c in self.cmds))

    @property
    def n_commands(self) -> int:
        return int(np.count_nonzero(self.cmds))

    def placements(self) -> List[Tuple[object, int]]:
        """[(slot, command), ...] for every placed command, in board order."""
        return [(s, int(self.cmds[s.index])) for s in self.board.slots if self.cmds[s.index]]

    def copy(self) -> "Plan":
        return Plan(self.board, self.cmds)

    def __eq__(self, other):
        return isinstance(other, Plan) and self.board.signature() == other.board.signature() and \
            np.array_equal(self.cmds, other.cmds)

    # ------------------------------------------------------------------ pretty printing
    def checklist(self) -> str:
        """One line per command, in the order you would click them into the game."""
        out = []
        for slot, cmd in self.placements():
            out.append(f"{slot.label:<34} {R.CMD_NAMES[cmd]}")
        out.append(f"{'':<34} total ${self.cost}")
        return "\n".join(out)

    def grid(self) -> str:
        """The board as a compact grid.  Each building shows ``top/bottom``; plants show ``P:x``.

        A dot means "leave that floor empty".  Arrows are turn commands, PU = Power Up, ST = Stomp.
        """
        b = self.board
        words = {R.T_GRASS: "grass", R.T_NONE: "--", R.T_MUD: "mud", R.T_SPIKE: "spikes", R.T_BOULDER: "boulder"}
        cells = []
        for r in range(R.ROWS):
            row = []
            for c in range(R.COLS):
                t = int(b.tiles[r, c])
                base = int(b.slot_base[r, c])
                if t in (R.T_LOW, R.T_HIGH):
                    row.append(f"{R.CMD_SHORT[int(self.cmds[base])]}/{R.CMD_SHORT[int(self.cmds[base + 1])]}")
                elif t == R.T_PLANT:
                    row.append(f"P:{R.CMD_SHORT[int(self.cmds[base])]}")
                elif t == R.T_BED:
                    row.append(f"{R.CAT_NAMES[int(b.bed_owner[r, c])]} bed")
                elif b.tokens[r][c].startswith("C"):
                    row.append(R.CAT_NAMES[R.CAT_LETTERS.index(b.tokens[r][c][1])])
                else:
                    row.append(words[t])
            cells.append(row)
        width = max(len(x) for row in cells for x in row) + 2
        head = "      " + "".join(f"c{c + 1}".ljust(width) for c in range(R.COLS))
        lines = [head]
        for r, row in enumerate(cells):
            lines.append(f"r{r + 1}".ljust(6) + "".join(x.ljust(width) for x in row))
        return "\n".join(lines)

    def __repr__(self):
        return f"Plan({self.board.name!r}, {self.n_commands} commands, ${self.cost})"
