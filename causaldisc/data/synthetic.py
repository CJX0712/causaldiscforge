"""Synthetic causal DAG generators for benchmarking.

Every generator returns a dataset dict with:
    name     : str
    X        : [n, d] observational data
    adj      : [d, d] int DAG adjacency (adj[i,j]==1 iff i -> j)
    W        : [d, d] float weight matrix (W[i,j] = effect of i on j)
    cpdag    : CPDAG (the ground-truth CPDAG)
    n, d     : shapes
    n_edges  : true edge count
    meta     : dict (description / difficulty note)

All DAGs are forward (edges only i<j) so a simple in-order SEM is valid.
Difficulty knobs (SNR ~ 1.0, n/d^2 ~ 5, avg degree ~ 2) are tuned so that
baselines score a *non-trivial* SHD (not 0, not fully broken).
"""
from __future__ import annotations

import numpy as np

from causaldisc.core.types import CPDAG
from causaldisc.graph.primitives import dag_to_cpdag


def _simulate(adj: np.ndarray, W: np.ndarray, funcs=None, sigma=1.0, n=500, seed=0):
    rng = np.random.default_rng(seed)
    d = adj.shape[0]
    X = rng.normal(0.0, sigma, size=(n, d))
    for j in range(d):
        for p in range(j):  # parents already final (forward DAG)
            if adj[p, j] == 1:
                coeff = W[p, j]
                if funcs is not None and (j, p) in funcs:
                    X[:, j] += coeff * funcs[(j, p)](X[:, p])
                else:
                    X[:, j] += coeff * X[:, p]
    return X


def _from_edges(d, edges, weights_range, seed=0, n=500, sigma=1.0, funcs=None):
    rng = np.random.default_rng(seed)
    adj = np.zeros((d, d), dtype=int)
    W = np.zeros((d, d), dtype=float)
    for (a, b) in edges:
        adj[a, b] = 1
        lo, hi = weights_range
        W[a, b] = float(rng.uniform(lo, hi) * (1 if rng.random() > 0.5 else -1))
    X = _simulate(adj, W, funcs=funcs, sigma=sigma, n=n, seed=seed)
    return X, adj, W


def chain_dgp(d=10, n=500, seed=0):
    edges = [(i, i + 1) for i in range(d - 1)]
    X, adj, W = _from_edges(d, edges, (0.5, 1.0), seed=seed, n=n)
    return _pack("chain", X, adj, W, "long-range partial correlations (PC depth)", n, d)


def fork_dgp(d=8, n=500, seed=0):
    edges = [(0, j) for j in range(1, d)]
    X, adj, W = _from_edges(d, edges, (0.6, 1.0), seed=seed, n=n)
    return _pack("fork", X, adj, W, "confounder vs v-structure confusion", n, d)


def collider_dgp(d=10, n=1000, seed=0):
    edges = [(0, 2), (1, 2), (3, 5), (4, 5), (6, 8), (7, 8)]
    # keep within d=10
    edges = [(min(a, d - 1), min(b, d - 1)) for a, b in edges if b < d]
    X, adj, W = _from_edges(d, edges, (0.5, 1.0), seed=seed, n=n)
    return _pack("collider", X, adj, W, "orientation only (3 v-structures)", n, d)


def tree_dgp(d=20, n=500, seed=0):
    rng = np.random.default_rng(seed)
    edges = []
    for j in range(1, d):
        p = int(rng.integers(0, j))  # random parent -> arborescence
        edges.append((p, j))
    X, adj, W = _from_edges(d, edges, (0.4, 0.9), seed=seed, n=n)
    return _pack("tree", X, adj, W, "scale d=20, max-level stress", n, d)


def dense_dgp(d=10, n=1000, seed=0, p=0.4):
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < p:
                edges.append((i, j))
    # denser + weaker-weight graph so that PC (max_level=3) can neither recover
    # every long-range independence nor fully orient -> non-trivial SHD (not 0).
    X, adj, W = _from_edges(d, edges, (0.25, 0.5), seed=seed, n=n)
    return _pack("dense", X, adj, W, "NOTEARS threshold sensitivity (weak edges)", n, d)


def hard_dgp(d=12, n=1000, seed=0, n_edges=15):
    rng = np.random.default_rng(seed)
    funcs = {}
    edges = []
    tries = 0
    while len(edges) < n_edges and tries < 2000:
        tries += 1
        a = int(rng.integers(0, d - 1))
        b = int(rng.integers(a + 1, d))
        if (a, b) in edges:
            continue
        edges.append((a, b))
        kind = int(rng.integers(0, 3))
        if kind == 0:
            funcs[(b, a)] = lambda x: np.tanh(x) / 2.0
        elif kind == 1:
            funcs[(b, a)] = lambda x: (x ** 2) / 4.0
        else:
            funcs[(b, a)] = lambda x: np.sin(x) / 3.0
    X, adj, W = _from_edges(d, edges, (0.5, 1.0), seed=seed, n=n, sigma=1.0, funcs=funcs)
    # heteroskedastic noise
    rng2 = np.random.default_rng(seed + 1)
    noise = rng2.normal(0.0, 1.0, size=X.shape)
    scale = 0.5 + 0.5 * np.abs(X)
    X = X + scale * noise
    return _pack("hard", X, adj, W, "nonlinear + heteroskedastic (assumption violation)", n, d)


def _pack(name, X, adj, W, meta, n, d):
    cpdag = dag_to_cpdag(adj)
    return {
        "name": name,
        "X": X,
        "adj": adj,
        "W": W,
        "cpdag": cpdag,
        "n": n,
        "d": d,
        "n_edges": int(np.sum(adj)),
        "meta": meta,
    }


def load_all(seeds=None):
    """Return all 6 datasets (defaults: each with a fixed seed)."""
    if seeds is None:
        seeds = {
            "chain": 11, "fork": 22, "collider": 33,
            "tree": 44, "dense": 55, "hard": 66,
        }
    return [
        chain_dgp(seed=seeds["chain"]),
        fork_dgp(seed=seeds["fork"]),
        collider_dgp(seed=seeds["collider"]),
        tree_dgp(seed=seeds["tree"]),
        dense_dgp(seed=seeds["dense"]),
        hard_dgp(seed=seeds["hard"]),
    ]
