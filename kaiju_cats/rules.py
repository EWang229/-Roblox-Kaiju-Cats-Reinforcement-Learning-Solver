"""
Game constants and rule switches for Kaiju Cats.

Everything that the simulators need to agree on lives here, so there is exactly one place to look
when the game turns out to behave differently from what we assumed.

Rule status legend (see docs/INSIGHTS.md for the full discussion):
    CONFIRMED  - checked against a real in-game replay, or stated in the in-game instructions
    ASSUMED    - our best reading; never contradicted so far, but never directly tested either
"""
from dataclasses import dataclass

# --------------------------------------------------------------------------------------------
# Board geometry
# --------------------------------------------------------------------------------------------
ROWS, COLS = 5, 7

# Directions are indexed 0..3 = North, East, South, West.  (dr, dc) is the step in (row, col).
NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3
DR = (-1, 0, 1, 0)
DC = (0, 1, 0, -1)
DIR_NAMES = ("North", "East", "South", "West")
DIR_ARROWS = ("↑", "→", "↓", "←")  # up, right, down, left arrows

# --------------------------------------------------------------------------------------------
# Tiles.  The text board format uses two-character tokens (see board.py).
# --------------------------------------------------------------------------------------------
T_NONE = 0     # "--"  water / off-board; cats bounce off it like a wall
T_GRASS = 1    # ".."  empty tile (cat start tiles are grass too)
T_LOW = 2      # "GB"  gray building:   2 floors worth 250 each
T_HIGH = 3     # "YB"  yellow building: 2 floors worth 500 each
T_PLANT = 4    # "PP"  power plant:     1 floor that DOUBLES the cat's power
T_MUD = 5      # "MD"  mud: holds a cat for one extra turn
T_SPIKE = 6    # "SP"  spikes: halve the cat's power
T_BOULDER = 7  # "RK"  boulder: reverses the cat and costs the turn
T_BED = 8      # "bB", "bR", "bG": a cat's bed (only its owner may enter)

FLOOR_VALUE = {T_LOW: 250, T_HIGH: 500}   # plants are multiplicative, not additive

# --------------------------------------------------------------------------------------------
# Commands.  A command is attached to ONE floor of a building (or to a plant) and fires when a
# cat destroys that floor.  After firing it is consumed.
# --------------------------------------------------------------------------------------------
CMD_NONE = 0
CMD_NORTH, CMD_EAST, CMD_SOUTH, CMD_WEST = 1, 2, 3, 4   # "turn" commands: direction = cmd - 1
CMD_STOMP = 5      # the cat stays and destroys the next floor of the same building next turn
CMD_POWER_UP = 6   # +1000 power on top of the floor's own value
N_COMMANDS = 7
CMD_COST = (0, 10, 10, 10, 10, 20, 30)
CMD_NAMES = ("none", "North", "East", "South", "West", "Stomp", "Power Up")
CMD_SHORT = (".", "↑", "→", "↓", "←", "ST", "PU")

BUDGET = 200       # CONFIRMED: you may clear the pre-placed commands and spend $200
TURNS = 15         # CONFIRMED

# --------------------------------------------------------------------------------------------
# Cats and beds
# --------------------------------------------------------------------------------------------
CAT_NAMES = ("Blue", "Red", "Green")
CAT_LETTERS = "BRG"

# Bed arrival bonus, by order of arrival (CONFIRMED by in-game replays):
#   first cat in   -> +2000
#   second cat in  -> x3
#   third cat in   -> x5
# Cats arriving on the same turn enter lowest-power first (ties: lower bed first).
ARRIVAL_ADD_FIRST = 2000
ARRIVAL_MULT_SECOND = 3
ARRIVAL_MULT_THIRD = 5

POWER_UP_BONUS = 1000


@dataclass(frozen=True)
class Rules:
    """Switches for the rules we are not 100 percent sure about.  Defaults are the best current reading."""

    # CONFIRMED (in-game replay, Oct 2026): a cat that bounces off a boulder or the board edge comes back
    # onto the tile it was standing on in the same turn, and breaks that building's NEXT floor if one is
    # left.  The bounce still costs the turn.
    bounce_breaks_floor: bool = True

    # CONFIRMED indirectly (scores matched): hitting the board edge costs the turn.
    edge_costs_turn: bool = True

    # ASSUMED, never tested: a Power Up attached to a plant is added AFTER the doubling (0).
    # Set to 1 to model "add 1000, then double".
    plant_power_up_order: int = 0

    turns: int = TURNS
    budget: int = BUDGET

    def as_flags(self):
        """Compact int tuple handed to the numba kernels (they cannot take dataclasses)."""
        # last entry = "solo cat" (-1 = normal game); the solver sets it for single-cat relaxations
        return (int(self.bounce_breaks_floor), int(self.edge_costs_turn), int(self.plant_power_up_order),
                int(self.turns), int(self.budget), -1)
