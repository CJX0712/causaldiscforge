"""Optuna HPO for NOTEARS lambda / w-threshold, optimised on BIC (no GT needed)."""
from __future__ import annotations

import numpy as np

from causaldisc.core.config import Config
from causaldisc.discover.notears import notears
from causaldisc.graph.primitives import bic_of_cpdag


def tune_notears(X, config: Config | None = None, n_trials: int | None = None):
    import optuna

    config = config or Config()
    n_trials = n_trials or config.hpo_trials
    X = np.asarray(X, float)

    def objective(trial):
        lam = trial.suggest_float("lambda", 0.01, 0.2, log=True)
        w_thr = trial.suggest_float("w_thr", 0.05, 0.4)
        cp = notears(X, lambdas=(lam,), w_thresholds=(w_thr,),
                     h_max_iter=100, seed=config.random_seed)
        return bic_of_cpdag(X, cp)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
