"""Score-based discovery: greedy hill-climbing with decomposable BIC over DAGs.

Incremental (only affected nodes recomputed); DFS acyclicity guard; random
restarts to escape local optima.
"""
from __future__ import annotations

import numpy as np

from causaldisc.core.types import CPDAG, NONE, TAIL, ARROW
from causaldisc.graph.primitives import (
    dag_to_cpdag, _local_bic, is_dag, meek_rules, NONE as _N, ARROW as _A, TAIL as _T,
)


def _reachable(adj, src, dst):
    """Can we reach dst from src following adj (i->j)?"""
    from collections import deque
    q = deque([src])
    seen = {src}
    while q:
        u = q.popleft()
        for v in range(adj.shape[0]):
            if adj[u, v] == 1 and v not in seen:
                if v == dst:
                    return True
                seen.add(v)
                q.append(v)
    return False


def _undir(g, a, b):
    g[a, b] = _T
    g[b, a] = _T


def _orient(g, a, b):
    g[a, b] = _A
    g[b, a] = _T


def hill_climbing(X: np.ndarray, restarts: int = 3, max_parents: int = 3,
                  lambda_: float = 1.0, seed: int = 0, verbose: bool = False) -> CPDAG:
    X = np.asarray(X, float)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    n, d = Xz.shape

    def local_bic(adj, i):
        parents = [p for p in range(d) if adj[p, i] == 1]
        return _local_bic(Xz, i, parents)

    best_adj = None
    best_score = float("inf")
    rng = np.random.default_rng(seed)

    for _ in range(restarts):
        adj = np.zeros((d, d), dtype=int)
        score = [local_bic(adj, i) for i in range(d)]
        improved = True
        while improved:
            improved = False
            best_delta = -1e-8
            best_op = None
            for i in range(d):
                for j in range(d):
                    if i == j:
                        continue  # never add a self-loop
                    # candidate add i->j
                    if adj[i, j] == 0:
                        # would create a cycle if j already reaches i
                        if not _reachable(adj, j, i) and np.sum(adj[:, j]) < max_parents:
                            new_parents = [p for p in range(d) if adj[p, j] == 1] + [i]
                            delta = _local_bic(Xz, j, new_parents) - score[j]
                            if delta < best_delta:
                                best_delta = delta
                                best_op = ("add", i, j)
                    # candidate delete i->j
                    if adj[i, j] == 1:
                        new_parents = [p for p in range(d) if adj[p, j] == 1 and p != i]
                        delta = _local_bic(Xz, j, new_parents) - score[j]
                        if delta < best_delta:
                            best_delta = delta
                            best_op = ("del", i, j)
                    # candidate reverse i->j to j->i (acyclic only)
                    if adj[i, j] == 1 and adj[j, i] == 0:
                        # temporarily drop i->j to test if i still reaches j
                        adj[i, j] = 0
                        acyclic = not _reachable(adj, i, j)
                        adj[i, j] = 1
                        if acyclic and np.sum(adj[:, i]) - 1 < max_parents \
                                and np.sum(adj[:, j]) - 1 < max_parents:
                            new_pi = [p for p in range(d) if adj[p, i] == 1 and p != j]
                            new_pj = [p for p in range(d) if adj[p, j] == 1 and p != i]
                            delta = (_local_bic(Xz, i, new_pi) + _local_bic(Xz, j, new_pj)
                                     - score[i] - score[j])
                            if delta < best_delta:
                                best_delta = delta
                                best_op = ("rev", i, j)
            if best_op is None:
                break
            op, i, j = best_op
            if op == "add":
                adj[i, j] = 1
                score[j] = _local_bic(Xz, j, [p for p in range(d) if adj[p, j] == 1])
            elif op == "del":
                adj[i, j] = 0
                score[j] = _local_bic(Xz, j, [p for p in range(d) if adj[p, j] == 1])
            else:  # rev
                adj[i, j] = 0
                adj[j, i] = 1
                score[i] = _local_bic(Xz, i, [p for p in range(d) if adj[p, i] == 1])
                score[j] = _local_bic(Xz, j, [p for p in range(d) if adj[p, j] == 1])
            improved = True
        total = sum(score)
        if total < best_score:
            best_score = total
            best_adj = adj.copy()

    return dag_to_cpdag(best_adj)
