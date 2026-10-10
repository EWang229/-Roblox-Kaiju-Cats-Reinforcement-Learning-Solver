# kaiju-cats-solver

A simulator and optimiser for the **Kaiju Cats** practice puzzle: three cats, a 5x7 map of buildings, $200 and 15 turns. Pick where to put commands so the cats smash as much as possible and reach their beds.

> \*\*Practice and learning only.\*\* This repo works on the public practice game. It does not read the screen, press keys or play for you, and it should not be used on a live, scored assessment.

## Quick Explanation

**The map decides what is possible.** Every board has its own best score, and across the 8 boards here it ranges from about 86k to 224k with the same money and turns. So judge a plan against the best known score for *that* board, not against a number from a different one. Details and tips are in [docs/INSIGHTS.md](docs/INSIGHTS.md).

## Install

```bash
pip install -r requirements.txt      # numpy, numba, joblib, pandas, matplotlib, tqdm, gymnasium, optuna, torch, pytest
python -m pytest                     # includes a replay of a real game and the RL environment checks
```

## Use

```bash
# known boards answer instantly from the solution library
python -m kaiju\_cats solve boards/board\_02.txt

# a new board: write it as 5 lines of 7 tokens (see below), then search for a few minutes
python -m kaiju\_cats solve my\_board.txt --preset careful --minutes 10 --plot plan.png

# check a plan you already have
python -m kaiju\_cats show boards/board\_02.txt tests/data/replay\_board2.plan

python -m kaiju\_cats library                          # what is already solved
python -m kaiju\_cats random-board --seed 3            # a fresh board to practise on
```

Board tokens: `CB`/`CR`/`CG` cat starts (Blue, Red, Green), `bB`/`bR`/`bG` beds, `GB` gray building (2 x 250), `YB` yellow building (2 x 500), `PP` plant (x2), `MD` mud, `SP` spikes, `RK` boulder, `..` empty, `--` unused.

Plans are printed as a map (`top/bottom` command per building, `P:` for plants, `.` for no command, `PU` = Power Up, `ST` = Stomp) and as a checklist of every command with its row, column and floor.

In Python:

```python
from kaiju\_cats import Board, SolveConfig, solve, simulate
board = Board.from\_file("boards/board\_02.txt")
result = solve(board, SolveConfig.preset("fast"))
print(result.summary())
print(result.plan.checklist())
```

## How the solver works

1. **Lazy decisions.** A command only matters when its floor is destroyed. The game is played forward and commands are decided at that moment, so plans never contain dead commands.
2. **Parallel tempering.** Several chains at different temperatures explore plans and swap. One group of chains is run for each of the 6 possible bed-arrival orders (`fastsim.py`, compiled with numba; `solver.py`, run in parallel with joblib).
3. **Exact local polish.** After the search, a depth-first search re-routes one cat (or two) exactly while the rest stay fixed (`lazy\_dfs.py`), plus exhaustive small-change polishing.
4. **Many seeds.** Success in the search is a lottery, so `--minutes` re-solves with fresh seeds and keeps the best.
5. **Everything is verified** by a second, readable simulator (`simulator.py`) before it is reported.

The ML/data-science packages are used where they help: numba (speed), joblib (parallel chains), pandas and matplotlib (reporting), PyTorch (the policy network), Optuna (hyper-parameter tuning, wired in but not yet run) and Gymnasium (`env.py`).

## Results

Python `careful` preset, 3 seeds (12, 13, 14), about 2.5-3 minutes per solve on a 2-core machine. "Best known" is the best plan I found over many runs of the original C++ solver; every plan in the library is re-verified by the Python simulator.

|Board|Best known|Python, best of 3 seeds|Python, mean|Python / best known|
|-|-:|-:|-:|-:|
|board\_01|85,875|85,875|85,750|100.0%|
|board\_02|193,375|182,406|179,489|94.3%|
|board\_03|219,375|219,375|216,458|100.0%|
|board\_04|214,375|139,875|139,875|65.2%|
|board\_05|177,875|176,125|173,208|99.0%|
|board\_06|136,125|133,875|133,875|98.3%|
|board\_07|115,750|115,250|106,417|99.6%|
|board\_08|223,750|223,750|214,958|100.0%|

What this says, plainly:

* On 3 of 8 boards the Python solver matched the best known score in at least one seed (boards 1, 3 and 8), and it was within 3% on boards 5, 6 and 7.
* It is **not** as strong as the C++ solver on hard boards. Board 4 never left 139,875 (best known 214,375), and board 2 reached 182,406 against 193,375. One run is a lottery: board 8 gave 223,750, 223,750 and 197,375.
* In total the best-of-3 Python scores add up to 93.4% of the best-known scores.
* Use `--minutes` for more seeds on a board that matters, and store the best result in the library (`solve` does it automatically).
* The best known scores are the best plans found so far, not proven optimal.

## Reinforcement learning

A small neural-network policy (84k parameters) was trained with **expert iteration** on randomly generated boards: the policy plays, a fast local search improves its best games, and the policy learns to imitate the improved games. Given a new board, `rl-solve` samples thousands of games, polishes the best ones, adapts to the board, and verifies the answer in the independent simulator, all in about two minutes.

```bash
python -m kaiju\_cats rl-solve boards/board\_04.txt --seconds 120
python scripts/train\_exit.py --iterations 700 --out runs/policy.pt     # retrain (about 90 min on 2 CPU cores)
```

On the 8 real boards (never trained on), one 2-minute run reached **96-98% of the best-known total** over three seeds, and matched the best known score on every seed on 4 boards (it found 214,375 on board 4, where my search solver stalled at 139,875). It never beat a best-known score, and it is weak on boards 2 and 5. On 8 held-out generated boards it matched or beat the search solver's reference on all 8. Full tables, the failures along the way (plain policy gradient learned too slowly), and the reasons behind every design decision are in [**docs/RL.md**](docs/RL.md).

Honesty note: part of the pipeline is classical search (the "expert"), so this is RL-guided search, not pure RL. The "policy sampling alone" column in docs/RL.md shows what the network does by itself (71% of the best-known total).

## Layout

```
kaiju\_cats/
  rules.py       constants and the Rules switches
  board.py       board parsing
  plan.py        plans, checklists
  simulator.py   readable reference simulator (use this to learn the rules)
  fastsim.py     numba simulator + tempering kernels
  lazy\_dfs.py    exact re-optimisation of one cat's route
  solver.py      the pipeline and presets
  library.py     solved boards (data/library.json)
  analysis.py render.py env.py generate.py benchmark.py tune.py cli.py
  rl/            batched environment, policy network, training (expert iteration), board-time solver
  data/policy.pt the trained policy (340 KB)
boards/          the 8 real boards seen so far
boards\_val/      8 generated boards used only for validation (never trained on)
tests/           pytest, including the real-game replay
docs/INSIGHTS.md tips, tricks, rule status
docs/RL.md       the reinforcement-learning work, with every decision explained
```

## Honest caveats

* The model was checked against one real-game replay. One rule (a Power Up on a plant is added after doubling) is assumed and untested.
* The Python solver is not guaranteed to reproduce the best-known scores in one run; see Results.
* "Best known" means the best plan found so far; a better one may exist.

MIT licensed.

