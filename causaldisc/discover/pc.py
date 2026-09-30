"""PC-stable: constraint-based discovery via Fisher-Z partial-correlation tests."""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import norm

from causaldisc.core.types import CPDAG, NONE, TAIL, ARROW
from causaldisc.graph.primitives import partial_corr, meek_rules


def _undir(g, a, b):
    g[a, b] = TAIL
    g[b, a] = TAIL


def _orient(g, a, b):
    g[a, b] = ARROW
    g[b, a] = TAIL


def pc_stable(X: np.ndarray, alpha: float = 0.01, max_level: int = 3,
              verbose: bool = False) -> CPDAG:
    X = np.asarray(X, float)
    n, d = X.shape
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    z_crit = float(norm.ppf(1.0 - alpha / 2.0))

    g = np.zeros((d, d), dtype=np.int8)
    for i in range(d):
        for j in range(i + 1, d):
            _undir(g, i, j)

    sep = {}  # (i,j) -> set S that made i,j independent

    for k in range(0, max_level + 1):
        cur = g.copy()  # PC-stable: snapshot within a level
        edges = [(i, j) for i in range(d) for j in range(i + 1, d) if cur[i, j] != NONE]
        for (i, j) in edges:
            neigh = [m for m in range(d) if cur[i, m] != NONE and m != j]
            for S in combinations(neigh, k):
                if n < len(S) + 10:  # hard freedegree guard
                    continue
                r = partial_corr(Xz, i, j, S)
                z = 0.5 * np.sqrt(max(n - len(S) - 3, 1)) * np.log((1 + abs(r)) / (1 - abs(r)))
                if abs(z) <= z_crit:
                    g[i, j] = NONE
                    g[j, i] = NONE
                    sep[(i, j)] = set(S)
                    sep[(j, i)] = set(S)
                    break

    # v-structures: non-adjacent i,j with common neighbour k, k NOT in sep(i,j)
    for i in range(d):
        for j in range(i + 1, d):
            if g[i, j] != NONE:
                continue
            for k in range(d):
                if k == i or k == j:
                    continue
                if g[i, k] != NONE and g[j, k] != NONE:
                    if k not in sep.get((i, j), set()):
                        _orient(g, i, k)
                        _orient(g, j, k)

    g = meek_rules(g)
    return CPDAG(d, g, name="pc")


def pc_oracle(adj_true: np.ndarray, X: np.ndarray, alpha: float = 0.01,
              max_level: int = 3) -> CPDAG:
    """Oracle PC: replaces the statistical CI test with d-separation on the true
    DAG. Used by invariant tests; must recover the true CPDAG (SHD=0)."""
    from causaldisc.graph.primitives import d_separation
    d = adj_true.shape[0]
    g = np.zeros((d, d), dtype=np.int8)
    for i in range(d):
        for j in range(i + 1, d):
            _undir(g, i, j)
    sep = {}
    for k in range(0, max_level + 1):
        cur = g.copy()
        edges = [(i, j) for i in range(d) for j in range(i + 1, d) if cur[i, j] != NONE]
        for (i, j) in edges:
            neigh = [m for m in range(d) if cur[i, m] != NONE and m != j]
            found = False
            for S in combinations(neigh, k):
                if d_separation(adj_true, i, j, S):
                    g[i, j] = NONE
                    g[j, i] = NONE
                    sep[(i, j)] = set(S)
                    sep[(j, i)] = set(S)
                    found = True
                    break
            if found:
                continue
    for i in range(d):
        for j in range(i + 1, d):
            if g[i, j] != NONE:
                continue
            for k in range(d):
                if k == i or k == j:
                    continue
                if g[i, k] != NONE and g[j, k] != NONE:
                    if k not in sep.get((i, j), set()):
                        _orient(g, i, k)
                        _orient(g, j, k)
    g = meek_rules(g)
    return CPDAG(d, g, name="pc_oracle")
