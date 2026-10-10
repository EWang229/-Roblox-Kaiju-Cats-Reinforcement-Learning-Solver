"""Pictures: a board with its plan drawn on it, plus the paths the cats take."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")           # render to files; works on servers without a display
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from . import rules as R
from .board import Board
from .plan import Plan
from .simulator import simulate

TILE_COLOR = {
    R.T_NONE: "#dfe9f2", R.T_GRASS: "#cfe7c3", R.T_LOW: "#d4d9df", R.T_HIGH: "#f1d98f",
    R.T_PLANT: "#cdb8ef", R.T_MUD: "#c9a982", R.T_SPIKE: "#a9b1b8", R.T_BOULDER: "#bcb5a9", R.T_BED: "#ffffff",
}
TILE_LABEL = {R.T_LOW: "gray", R.T_HIGH: "yellow", R.T_PLANT: "plant", R.T_MUD: "mud", R.T_SPIKE: "spikes",
              R.T_BOULDER: "boulder"}
CAT_COLOR = ("#2f6fe0", "#d83a3a", "#1f9d55")           # blue, red, green
GLYPH = {1: "↑", 2: "→", 3: "↓", 4: "←", 5: "ST", 6: "PU"}


def plot_plan(board: Board, plan: Plan, path: Optional[str] = None, paths: bool = True,
              title: Optional[str] = None, rules: R.Rules = R.Rules()):
    """Draw the board, the commands (top slot above bottom slot) and, optionally, each cat's route.

    Power Ups are drawn in gold so you can see at a glance where the $30 commands go.
    """
    result = simulate(board, plan, rules, trace=True)
    fig, ax = plt.subplots(figsize=(12, 8.6))
    ax.set_xlim(-0.1, R.COLS + 0.1)
    ax.set_ylim(R.ROWS + 0.1, -0.1)
    ax.set_aspect("equal")
    ax.axis("off")

    for r in range(R.ROWS):
        for c in range(R.COLS):
            t = int(board.tiles[r, c])
            if t == R.T_NONE:
                continue
            ax.add_patch(FancyBboxPatch((c + 0.04, r + 0.04), 0.92, 0.92, boxstyle="round,pad=0,rounding_size=0.08",
                                        fc=TILE_COLOR[t], ec="#56635d" if t != R.T_BED else CAT_COLOR[int(board.bed_owner[r, c])],
                                        lw=1.0 if t != R.T_BED else 3.0))
            tk = board.tokens[r][c]
            if tk[0] == "C":
                cat = R.CAT_LETTERS.index(tk[1])
                ax.text(c + 0.5, r + 0.5, f"{R.CAT_NAMES[cat]}\nstart", ha="center", va="center", fontsize=9,
                        color=CAT_COLOR[cat], fontweight="bold")
            elif t == R.T_BED:
                cat = int(board.bed_owner[r, c])
                ax.text(c + 0.5, r + 0.5, f"{R.CAT_NAMES[cat]}\nbed", ha="center", va="center", fontsize=9,
                        color=CAT_COLOR[cat], fontweight="bold")
            elif t in TILE_LABEL:
                ax.text(c + 0.5, r + 0.08, TILE_LABEL[t], ha="center", va="top", fontsize=7, color="#33403a")
            base = int(board.slot_base[r, c])
            if base >= 0:
                n = 2 if t in (R.T_LOW, R.T_HIGH) else 1
                for f in range(n):
                    cmd = int(plan.cmds[base + f])
                    y = r + (0.34 if n == 2 else 0.5) + f * 0.3
                    ax.add_patch(FancyBboxPatch((c + 0.17, y - 0.12), 0.66, 0.24, boxstyle="round,pad=0,rounding_size=0.06",
                                                fc="#ffcf33" if cmd == R.CMD_POWER_UP else ("#16201c" if cmd else "#ffffff"),
                                                ec="#8a6d00" if cmd == R.CMD_POWER_UP else "#9aa6a0", lw=0.8))
                    ax.text(c + 0.5, y, GLYPH.get(cmd, "·"), ha="center", va="center", fontsize=11,
                            color="#16201c" if cmd == R.CMD_POWER_UP else ("#ffffff" if cmd else "#9aa6a0"),
                            fontweight="bold")

    if paths:
        for cat in range(3):
            pts = [(board.starts[cat][1] + 0.5, board.starts[cat][0] + 0.5)]
            for snap in result.history:
                rr, cc = snap["cats"][cat][0], snap["cats"][cat][1]
                pts.append((cc + 0.5, rr + 0.5))
            off = (cat - 1) * 0.07
            xs = [p[0] + off for p in pts]
            ys = [p[1] + off for p in pts]
            ax.plot(xs, ys, "-", color=CAT_COLOR[cat], lw=2.2, alpha=0.55, zorder=5)
            ax.plot(xs[-1:], ys[-1:], "o", color=CAT_COLOR[cat], ms=6, zorder=6)

    cats = "   ".join(f"{R.CAT_NAMES[i]} {s:,}" for i, s in enumerate(result.cat_scores))
    ax.set_title(title or f"{board.name}: {result.score:,} points at ${plan.cost}\n{cats}", fontsize=12)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
    return fig
