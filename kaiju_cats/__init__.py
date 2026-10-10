"""kaiju_cats - a simulator and optimiser for the Kaiju Cats practice puzzle.

Quick start::

    from kaiju_cats import Board, Plan, simulate, solve, SolveConfig
    board = Board.from_file("boards/board_02.txt")
    result = solve(board, SolveConfig.preset("fast"))
    print(result.summary()); print(result.plan.checklist())
"""
from .board import Board
from .plan import Plan
from .rules import Rules
from .simulator import simulate
from .solver import SolveConfig, SolveResult, solve, solve_for

__all__ = ["Board", "Plan", "Rules", "simulate", "solve", "solve_for", "SolveConfig", "SolveResult"]
__version__ = "0.1.0"
