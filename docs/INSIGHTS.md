# What I learned about Kaiju Cats

Everything here comes from simulating the practice game and from replaying plans in it. Each claim is tagged:

* **Confirmed**: checked against the real game (a replay of a full plan matched the score exactly).
* **Measured**: a statistic from the plans in `kaiju_cats/data/library.json` (8 boards, so treat it as a pattern, not a law).
* **Assumed**: modelled that way but never tested in the game.

This repo is for practice and learning. Don't use it on a live, scored assessment.

---

## 1. The map decides what is possible

You choose commands, but the map decides how much those commands can be worth.

* The board is a fixed 5x7 layout. Which tiles sit on which route, where the plants (x2) are, where the spikes and mud sit, and which cat starts next to which bed all come with the map. None of that is yours to change.
* On the 8 boards in this repo the best plans I found range from **85,875 to 223,750**, a 2.6x spread, all with the same $200 and 15 turns.
* A weak result on one map is not necessarily a weak plan. Compare against the best known score for that board, not against a number from a different map.
* Boards repeat. Solved boards are in the library, so `solve` answers instantly for a board it has seen.

## 2. Where the points come from

* **One cat carries the score.** In all 8 best plans one cat holds 83-96% of the points (Measured). The other two cats exist mostly to clear the way, take fights early, and arrive first at their beds.
* **The big cat enters its bed last (x5).** Arrival bonuses are +2000 for the first cat, x3 for the second, x5 for the third. In all 8 best plans the big cat is the third to arrive (Measured), and arrives on turn 15 (Measured), so the last turn is used with no slack.
* **Plants double power.** A plant (one floor, x2) on the big cat's route is worth far more than a building floor. Hitting several plants in a row compounds.
* **Power Ups are +1000 for $30, and they are best used before a doubling.** The typical plan spends 3 Power Ups ($90) and about 11 turn commands (Measured). Putting a Power Up on a floor the big cat destroys *before* a plant or the x5 bed bonus multiplies that 1000.
* **The money is nearly always fully spent.** Every best plan costs exactly $200 (Measured). Leftover money is wasted points.

## 3. Tricks

* **Bounces are free floors (Confirmed).** A cat that hits a boulder or the edge turns around, loses the turn, and breaks the next floor of the building it is standing on, firing that floor's value and command. A bounce on a tall building is a free hit, and a well-placed rock can be used as a "stomp" without paying for one.
* **A command only matters when its floor is destroyed.** A command on a floor nobody breaks costs nothing and does nothing. The solver only stores commands that fire, which is also why plans are cheap to check by hand.
* **Use a small cat as a decoy.** Fights remove the loser, with ties going to the topmost bed. A weak cat that is going to lose anyway can clear a path, soak a fight, or sit on a spike tile instead of the big cat.
* **Spikes halve power.** They cost little if a cat with almost no power takes them, and cost a lot if the big cat does. Mud costs the cat a turn, which is the same as one less step, so it is worth routing around on the tight turn-15 route.
* **Same-turn arrivals:** the lower-power cat enters first. This can be used to choose who gets the x3 and who gets the x5 (from the game's rules text; not isolated by any replay).
* **Stomp ($20)** holds a cat on its tile to break the next floor. Only worth it when the extra floor is worth more than a step. It was used in only 2 of the 8 best plans (Measured).

## 4. What to look out for when you read a board

1. **Which cat can reach the most plants and tall (yellow) buildings in 14 steps?** That is the big-cat candidate. It is the single biggest decision.
2. **Count steps.** The big cat arrives on turn 15 with zero slack, so every step must be paid for in turns. A detour that looks lucrative may not fit.
3. **Check the bed order you need.** The big cat last, and the other two cats ahead of it, usually means the cats' beds and starts decide which one *can* be last.
4. **Look for rocks and edges next to buildings on the route.** They give free floors.
5. **Spikes and mud on the straight route** decide whether the straight path wins or a detour does.
6. **Leave no money unspent.**

## 5. Notes on the search

* **This is mostly a search problem.** The state space is small and the simulator is exact, so classical search does most of the work. A policy trained with expert iteration (docs/RL.md) turned out to be a useful starting point for that search: in two minutes per board it reached 96-98% of the best-known total on the 8 real boards. The ML-ecosystem packages are used where they help: numpy/numba for speed, joblib for parallel chains, pandas/matplotlib for reporting, PyTorch for the policy, Gymnasium for an RL-style environment, Optuna for hyper-parameter tuning (wired in, not yet run).
* **The method.** Because a command matters only when its floor is destroyed, the solver searches over *lazy* decisions. Parallel tempering (several chains at different temperatures that swap) explores plans, each chain conditioned on one of the 6 possible bed-arrival orders. After that, an exact depth-first search re-routes one cat (or two) at a time while the rest stay put.
* **Chain success is a lottery.** Roughly 3-5% of chains reach the best region on hard boards (Measured on the C++ implementation). Several independent seeds beat one long run. On a hard board the C++ solver's seeds ranged from 139,875 to 214,375.
* **What did not help.** Seeding the full search with the best solo route of one cat, and tuning the temperature ladder on a single board. The code for them was removed from the package. (Plain policy-gradient training from scratch also learned too slowly; expert iteration worked, see docs/RL.md.)
* **Verify everything.** Agreement between heuristics is not evidence. The only things trusted are the simulator checked against the game and the plans re-simulated end to end.

## 6. Rule status

The model was checked end to end against one real-game replay (board 2: Blue 3,500 + Red 145,000 + Green 9,000 = 157,500, reproduced exactly by both simulators). That replay exercises most rules at once, but it cannot isolate each of them.

| Rule | Status |
|---|---|
| 5x7 grid, cats start in column 1 facing East, beds in column 7 | Confirmed (replay) |
| 15 turns, $200; N/E/S/W $10, Stomp $20, Power Up $30 (+1000) | Confirmed (replay) |
| Gray 2x250, yellow 2x500, plant x2 on one floor | Confirmed (replay) |
| A bounce off a boulder or the edge returns the cat to its tile, which breaks its next floor | **Confirmed**; this was the rule the first model got wrong (155,250 vs 157,500) |
| Edge bounce costs the turn | Confirmed (replay) |
| Spikes halve power; mud holds a cat one turn | Rules text; exercised by the replay |
| Bed bonuses +2000 / x3 / x5; lower power enters first on a same-turn arrival | Rules text; tie rule not isolated |
| Fight: the lower power vanishes, ties go to the topmost bed | Rules text; not isolated |
| A Power Up on a plant is added after the doubling | **Assumed, untested**. `Rules(plant_power_up_order=1)` flips it. |

If a replay ever disagrees with the simulator, add the plan to `tests/data` and find which rule differs before trusting any score.
