"""NOTEARS (Zheng et al., NeurIPS 2018): continuous optimization DAG learning.

    min  0.5/n ||X - X W||_F^2 + lambda ||W||_1   s.t.  h(W) = tr(exp(W∘W)) - d = 0

Solved with augmented Lagrangian + L-BFGS-B. Pure numpy/scipy (no torch).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from causaldisc.core.types import CPDAG
from causaldisc.graph.primitives import dag_to_cpdag, is_dag, bic_of_dag


def _h(W: np.ndarray) -> float:
    d = W.shape[0]
    return float(np.trace(expm(W * W)) - d)


def _grad_h(W: np.ndarray) -> np.ndarray:
    return np.multiply(expm(W * W).T, 2.0 * W)


def _notears_single(X: np.ndarray, lam: float, w_thr: float,
                    h_max_iter: int = 100, seed: int = 0, verbose: bool = False) -> np.ndarray:
    n, d = X.shape
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((d, d)) * 0.01)
    rho, alpha = 1.0, 0.0
    h_prev = np.inf
    for t in range(h_max_iter):
        def obj(w_flat):
            Wm = w_flat.reshape(d, d)
            M = X @ Wm - X
            loss = 0.5 / n * np.sum(M ** 2) + lam * np.sum(np.abs(Wm))
            h = _h(Wm)
            return loss + (rho / 2.0) * h * h + alpha * h

        def grad(w_flat):
            Wm = w_flat.reshape(d, d)
            M = X @ Wm - X
            g = X.T @ M / n
            g = g + lam * np.sign(Wm)
            h = _h(Wm)
            g = g + (rho * h + alpha) * _grad_h(Wm)
            return g.ravel()

        res = minimize(obj, W.ravel(), jac=grad, method="L-BFGS-B",
                       options={"maxiter": 50, "ftol": 1e-6})
        W = res.x.reshape(d, d)
        h = _h(W)
        if h > 0.25 * h_prev:
            rho *= 10.0
        else:
            alpha += rho * h
            h_prev = h
        if h <= 1e-8:
            break
    W[np.abs(W) < w_thr] = 0.0
    return W


def _prune_to_dag(W: np.ndarray) -> np.ndarray:
    d = W.shape[0]
    adj = (np.abs(W) > 0).astype(int)
    while not is_dag(adj):
        # remove smallest |W| edge
        cand = [(np.abs(W[i, j]), i, j) for i in range(d) for j in range(d) if adj[i, j] == 1]
        if not cand:
            break
        cand.sort(key=lambda t: t[0])
        _, i, j = cand[0]
        adj[i, j] = 0
        W[i, j] = 0.0
    return W


def notears(X: np.ndarray, lambdas=(0.01, 0.05, 0.1, 0.2),
            w_thresholds=(0.1, 0.2, 0.3), h_max_iter: int = 100,
            seed: int = 0, verbose: bool = False) -> CPDAG:
    X = np.asarray(X, float)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    best_adj = None
    best_bic = float("inf")
    best_W = None
    for li, lam in enumerate(lambdas):
        for wi, w_thr in enumerate(w_thresholds):
            W = _notears_single(Xz, lam, w_thr, h_max_iter=h_max_iter, seed=seed + li * 7 + wi)
            W = _prune_to_dag(W)
            adj = (np.abs(W) > 0).astype(int)
            bic = bic_of_dag(Xz, adj)
            if bic < best_bic:
                best_bic = bic
                best_adj = adj
                best_W = W
    return dag_to_cpdag(best_adj)
