"""
Readable reference simulator for Kaiju Cats.

This file favours clarity over speed.  It is used for

  * checking plans (the final answer of every solve is re-played here),
  * tracing a plan turn by turn,
  * the Gymnasium environment (it can pause whenever a floor is destroyed and ask for a command),
  * the unit tests, which compare it with the numba kernel on thousands of random plans.

One turn
--------
 1. Every active cat decides where it wants to go (stuck cats stay, stomping cats stay, blocked cats
    turn around and stay).
 2. Cats that end the turn on the same tile fight: the higher power wins, ties go to the topmost bed.
    The loser vanishes.
 3. Cats move and the tile they land on takes effect: bed (queue for the arrival bonus), mud (stuck for one
    turn), spikes (power halved), building/plant with floors left (destroy the top floor).
    A cat that bounced and stands on a building with a floor left also destroys that floor
    (rule ``bounce_breaks_floor``).
 4. Arrival bonuses are paid: lowest power enters first (+2000 / x3 / x5).
 5. Every floor destroyed this turn pays out its value and then fires its command, in cat order
    Blue, Red, Green.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional

import numpy as np

from . import rules as R
from .board import Board

ACTIVE, STUCK, STOMPING, IN_BED, GONE = range(5)
STATUS_NAMES = ("active", "stuck", "stomping", "in bed", "gone")


class Event(NamedTuple):
    """A floor was destroyed and its command (if any) is about to fire."""
    cat: int
    slot: int      # index into board.slots
    row: int
    col: int
    tile: int


@dataclass
class CatState:
    row: int
    col: int
    direction: int = R.EAST      # every cat starts facing East
    status: int = ACTIVE
    power: int = 0


@dataclass
class Result:
    score: int
    cat_scores: List[int]
    cost: int                       # money actually spent by commands that fired
    arrival_order: List[int]        # cats in the order they entered their beds
    arrival_turn: List[Optional[int]]
    used_slots: List[int]           # slots whose command was consulted (their floor got destroyed)
    history: list = field(default_factory=list)   # one snapshot per turn, filled when trace=True
    log: list = field(default_factory=list)       # (turn, cat, slot, cmd) for each destroyed floor
    stats: dict = field(default_factory=dict)     # counters: bounces, spikes, mud, lost fights, ...


class Game:
    """Step-able game.  Typical use is ``Game(board).play(plan.cmds)``."""

    def __init__(self, board: Board, rules: R.Rules = R.Rules(), trace: bool = False):
        self.board, self.rules, self.trace = board, rules, trace
        self.cats = [CatState(*board.starts[i]) for i in range(3)]
        self.floors = {}          # (row, col) -> floors left
        for s in board.slots:
            self.floors[(s.row, s.col)] = 2 if s.tile in (R.T_LOW, R.T_HIGH) else 1
        self.turn = 0
        self.spent = 0
        self.arrival_order: List[int] = []
        self.arrival_turn: List[Optional[int]] = [None, None, None]
        self.pending: List[Event] = []
        self.history: list = []
        self.log: list = []
        self.used_slots: List[int] = []
        self.stats: Counter = Counter()

    # ------------------------------------------------------------------ helpers
    @property
    def over(self) -> bool:
        return self.turn >= self.rules.turns or all(c.status >= IN_BED for c in self.cats)

    def _blocked(self, cat: int, r: int, c: int):
        """(blocked, is_edge) for cat `cat` trying to enter (r, c)."""
        if not (0 <= r < R.ROWS and 0 <= c < R.COLS):
            return True, True
        t = int(self.board.tiles[r, c])
        if t == R.T_NONE:
            return True, True
        if t == R.T_BOULDER:
            return True, False
        if t == R.T_BED and int(self.board.bed_owner[r, c]) != cat:
            return True, False
        return False, False

    def _floor_event(self, cat: int, r: int, c: int) -> Event:
        """Destroy the top remaining floor of the building at (r, c) and report which slot that was."""
        total = 2 if int(self.board.tiles[r, c]) in (R.T_LOW, R.T_HIGH) else 1
        left = self.floors[(r, c)]
        slot = int(self.board.slot_base[r, c]) + (total - left)
        self.floors[(r, c)] = left - 1
        return Event(cat, slot, r, c, int(self.board.tiles[r, c]))

    # ------------------------------------------------------------------ one turn of movement
    def _move_phase(self) -> List[Event]:
        b, rl, cats = self.board, self.rules, self.cats
        alive = [c.status < IN_BED for c in cats]
        target = [(c.row, c.col) for c in cats]
        moving = [False] * 3
        bounced = [False] * 3

        # 1. intentions
        for i, c in enumerate(cats):
            if not alive[i]:
                continue
            if c.status == STUCK:            # mud: skip this turn, then carry on in the same direction
                c.status = ACTIVE
                continue
            if c.status == STOMPING:         # stays put; handled below
                continue
            nr, nc = c.row + R.DR[c.direction], c.col + R.DC[c.direction]
            blocked, edge = self._blocked(i, nr, nc)
            if blocked:
                c.direction = (c.direction + 2) % 4          # turn around; the turn is spent
                bounced[i] = True
                self.stats["edge_bounces" if edge else "boulder_or_bed_bounces"] += 1
                if edge and not rl.edge_costs_turn:          # alternative rule: edge bounce is free
                    nr, nc = c.row + R.DR[c.direction], c.col + R.DC[c.direction]
                    if not self._blocked(i, nr, nc)[0]:
                        target[i], moving[i] = (nr, nc), True
            else:
                target[i], moving[i] = (nr, nc), True

        # 2. fights between cats that end on the same tile (powers are read BEFORE anything moves)
        lost = [False] * 3
        for i in range(3):
            for j in range(3):
                if i != j and alive[i] and alive[j] and target[i] == target[j]:
                    if cats[j].power > cats[i].power or \
                            (cats[j].power == cats[i].power and b.prio[j] > b.prio[i]):
                        lost[i] = True
        for i in range(3):
            if lost[i]:
                cats[i].status, cats[i].power, alive[i] = GONE, 0, False
                self.stats["fights_lost"] += 1

        # 3. move + landing effects
        events: List[Event] = []
        arrivals = []
        for i, c in enumerate(cats):
            if not alive[i]:
                continue
            if moving[i]:
                c.row, c.col = target[i]
                t = int(b.tiles[c.row, c.col])
                if t == R.T_BED:
                    arrivals.append(i)
                elif t == R.T_MUD:
                    c.status = STUCK
                    self.stats["mud_hits"] += 1
                elif t == R.T_SPIKE:
                    c.power //= 2
                    self.stats["spike_hits"] += 1
                elif t in (R.T_LOW, R.T_HIGH, R.T_PLANT) and self.floors[(c.row, c.col)] > 0:
                    events.append(self._floor_event(i, c.row, c.col))
            elif c.status == STOMPING:
                if self.floors.get((c.row, c.col), 0) > 0:
                    events.append(self._floor_event(i, c.row, c.col))
                else:
                    c.status = ACTIVE
            elif bounced[i] and rl.bounce_breaks_floor and self.floors.get((c.row, c.col), 0) > 0:
                events.append(self._floor_event(i, c.row, c.col))
                self.stats["bounce_floors"] += 1

        # 4. bed arrivals: lowest power first, ties -> lower bed first
        arrivals.sort(key=lambda i: (cats[i].power, b.prio[i]))
        for i in arrivals:
            rank = len(self.arrival_order)
            if rank == 0:
                cats[i].power += R.ARRIVAL_ADD_FIRST
            elif rank == 1:
                cats[i].power *= R.ARRIVAL_MULT_SECOND
            else:
                cats[i].power *= R.ARRIVAL_MULT_THIRD
            cats[i].status = IN_BED
            self.arrival_order.append(i)
            self.arrival_turn[i] = self.turn
        return events

    # ------------------------------------------------------------------ applying a command
    def apply(self, event: Event, cmd: int) -> None:
        """Pay out the destroyed floor, then fire `cmd`."""
        c = self.cats[event.cat]
        if c.status == STOMPING:
            c.status = ACTIVE
        self.spent += R.CMD_COST[cmd]
        if event.tile == R.T_PLANT:
            if self.rules.plant_power_up_order == 0:      # double, then add 1000
                c.power *= 2
                if cmd == R.CMD_POWER_UP:
                    c.power += R.POWER_UP_BONUS
            else:                                           # add 1000, then double
                if cmd == R.CMD_POWER_UP:
                    c.power += R.POWER_UP_BONUS
                c.power *= 2
        else:
            c.power += R.FLOOR_VALUE[event.tile]
            if cmd == R.CMD_POWER_UP:
                c.power += R.POWER_UP_BONUS
        if R.CMD_NORTH <= cmd <= R.CMD_WEST:
            c.direction = cmd - 1
        elif cmd == R.CMD_STOMP and self.floors[(event.row, event.col)] > 0:
            c.status = STOMPING
        self.log.append((self.turn, event.cat, event.slot, cmd))
        self.used_slots.append(event.slot)

    # ------------------------------------------------------------------ stepping API
    def advance(self) -> List[Event]:
        """Run turns until a floor gets destroyed (returns the events of that turn) or the game ends."""
        while not self.pending and not self.over:
            self.turn += 1
            self.pending = self._move_phase()
            if not self.pending:
                self._snapshot()
        return self.pending

    def decide(self, cmd: int) -> Optional[Event]:
        """Fire `cmd` for the oldest undecided event.  Returns that event."""
        event = self.pending.pop(0)
        self.apply(event, cmd)
        if not self.pending:
            self._snapshot()
        return event

    def _snapshot(self) -> None:
        if self.trace:
            self.history.append({
                "turn": self.turn,
                "cats": [(c.row, c.col, c.direction, c.status, c.power) for c in self.cats],
            })

    def finish(self) -> Result:
        scores = [0 if c.status == GONE else c.power for c in self.cats]
        return Result(sum(scores), scores, self.spent, list(self.arrival_order), list(self.arrival_turn),
                      sorted(set(self.used_slots)), self.history, self.log, dict(self.stats))

    # ------------------------------------------------------------------ convenience
    def play(self, cmds) -> Result:
        """Play a whole plan (array of commands, one per slot) and return the Result."""
        while True:
            events = self.advance()
            if not events:
                break
            while self.pending:
                self.decide(int(cmds[self.pending[0].slot]))
        return self.finish()


def simulate(board: Board, plan, rules: R.Rules = R.Rules(), trace: bool = False) -> Result:
    """Play `plan` (a Plan or an array of commands) on `board` with the readable simulator."""
    cmds = plan.cmds if hasattr(plan, "cmds") else np.asarray(plan)
    return Game(board, rules, trace=trace).play(cmds)
