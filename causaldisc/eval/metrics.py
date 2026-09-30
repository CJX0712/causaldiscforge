"""Evaluation metrics wrapper."""
from __future__ import annotations

import numpy as np

from causaldisc.core.types import CPDAG
from causaldisc.graph.primitives import (
    shd_cpdag, shd_dag, skeleton_f1, orientation_scores, bic_of_cpdag,
)


def evaluate(true_cpdag: CPDAG, est_cpdag: CPDAG, X: np.ndarray | None = None) -> dict:
    tg, eg = true_cpdag.graph, est_cpdag.graph
    p, r, f1 = orientation_scores(tg, eg)
    res = {
        "shd_cpdag": int(shd_cpdag(tg, eg)),
        "skeleton_f1": round(float(skeleton_f1(tg, eg)), 4),
        "orient_precision": round(float(p), 4),
        "orient_recall": round(float(r), 4),
        "orient_f1": round(float(f1), 4),
        "n_true_edges": int(true_cpdag.n_edges()),
        "n_est_edges": int(est_cpdag.n_edges()),
    }
    if X is not None:
        res["bic"] = round(float(bic_of_cpdag(X, est_cpdag)), 2)
    return res


def evaluate_dag(true_adj: np.ndarray, est_adj: np.ndarray) -> dict:
    return {"shd_dag": int(shd_dag(true_adj, est_adj))}
