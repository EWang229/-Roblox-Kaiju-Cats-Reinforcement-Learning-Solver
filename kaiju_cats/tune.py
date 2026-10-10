"""Tune the parallel-tempering hyper-parameters with Optuna.

The objective is the *mean polished score of a handful of fixed-length chains* on one board - noisy, so
treat the result as a hint, not as truth.  (In the experiments recorded in docs/INSIGHTS.md the default
settings were already near a plateau and tuning on one board did not transfer reliably to others.)
"""
from __future__ import annotations

from dataclasses import replace

from .benchmark import chain_scores
from .board import Board
from .solver import SolveConfig


def tune(board: Board, trials: int = 20, seconds: float = 10.0, chains: int = 6, seed: int = 0):
    import optuna

    base = SolveConfig.preset("careful")

    def objective(trial: "optuna.Trial") -> float:
        cfg = replace(
            base,
            n_replicas=trial.suggest_int("n_replicas", 6, 20),
            t_low=trial.suggest_float("t_low", 10, 200, log=True),
            t_high=trial.suggest_float("t_high", 2000, 20000, log=True),
            steps_per_sweep=trial.suggest_int("steps_per_sweep", 100, 1000, step=100),
            order_penalty=trial.suggest_float("order_penalty", 5000, 60000, log=True),
            p_swap_move=trial.suggest_int("p_swap_move", 0, 50, step=10),
        )
        scores = chain_scores(board, cfg, chains, seconds, seed=seed)
        return sum(scores) / len(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    print("best mean chain score:", round(study.best_value))
    print("best parameters:", study.best_params)
    return study
