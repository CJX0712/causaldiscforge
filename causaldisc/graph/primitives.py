"""Graph primitives: partial correlation, Meek rules, DAG->CPDAG, SHD (correct
un-ordered-pair accounting), BIC (decomposable), d-separation, acyclicity."""
from __future__ import annotations

import numpy as np

from causaldisc.core.types import NONE, TAIL, ARROW, CIRCLE, CPDAG

_TOL = 1e-12


def _ridge_cov(X: np.ndarray) -> np.ndarray:
    R = np.corrcoef(X, rowvar=False)
    R = np.asarray(R, dtype=float)
    R = (R + R.T) / 2.0
    R.flat[:: R.shape[0] + 1] += 1e-8  # ridge for stability
    return R


def partial_corr(X: np.ndarray, i: int, j: int, S: tuple) -> float:
    """Partial correlation of columns i and j given set S, via the precision matrix."""
    cols = [i, j] + list(S)
    sub = X[:, cols]
    R = _ridge_cov(sub)
    try:
        P = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        P = np.linalg.pinv(R)
    a, b = cols.index(i), cols.index(j)
    denom = np.sqrt(max(P[a, a], 1e-12) * max(P[b, b], 1e-12))
    r = -P[a, b] / denom
    return float(np.clip(r, -1.0, 1.0))


# ---------- CPDAG matrix helpers (endpoint-code convention) ----------

def _orient(g: np.ndarray, a: int, b: int) -> None:
    g[a, b] = ARROW
    g[b, a] = TAIL


def _undir(g: np.ndarray, a: int, b: int) -> None:
    g[a, b] = TAIL
    g[b, a] = TAIL


def _is_arrow(g: np.ndarray, a: int, b: int) -> bool:
    return g[a, b] == ARROW and g[b, a] == TAIL


def _is_undir(g: np.ndarray, a: int, b: int) -> bool:
    return g[a, b] == TAIL and g[b, a] == TAIL


def _adj(g: np.ndarray, a: int, b: int) -> bool:
    return g[a, b] != NONE


def meek_rules(g: np.ndarray) -> np.ndarray:
    """Apply Meek R1-R3 iteratively until fix-point. Returns a *copy* of g."""
    d = g.shape[0]
    g = g.copy()
    changed = True
    guard = 0
    while changed and guard < d * d + 10:
        changed = False
        guard += 1
        # R1: a->b--c and a,c not adjacent  =>  b->c
        for a in range(d):
            for b in range(d):
                if not _is_arrow(g, a, b):
                    continue
                for c in range(d):
                    if c in (a, b) or not _is_undir(g, b, c):
                        continue
                    if not _adj(g, a, c):
                        _orient(g, b, c)
                        changed = True
        # R2: a->b->c and a--c undirected  =>  a->c
        for a in range(d):
            for b in range(d):
                if not _is_arrow(g, a, b):
                    continue
                for c in range(d):
                    if c in (a, b) or not _is_arrow(g, b, c):
                        continue
                    if _is_undir(g, a, c):
                        _orient(g, a, c)
                        changed = True
        # R3 (collider): unshielded a-b-c with d->a and d->c and d not adj b => a->b, c->b
        for a in range(d):
            for b in range(d):
                if not _is_undir(g, a, b):
                    continue
                for c in range(d):
                    if c in (a, b) or not _is_undir(g, b, c) or _adj(g, a, c):
                        continue
                    for d_ in range(d):
                        if d_ in (a, b, c):
                            continue
                        if _is_arrow(g, d_, a) and _is_arrow(g, d_, c) and not _adj(g, d_, b):
                            _orient(g, a, b)
                            _orient(g, c, b)
                            changed = True
                            break
    return g


def dag_to_cpdag(adj: np.ndarray) -> CPDAG:
    """Convert a DAG adjacency (adj[i,j]==1 iff i->j) to its CPDAG.

    Steps: skeleton; orient v-structures (colliders that are unshielded);
    propagate with Meek R1-R3.
    """
    adj = np.asarray(adj, dtype=int)
    d = adj.shape[0]
    skeleton = ((adj + adj.T) > 0).astype(int)
    g = np.zeros((d, d), dtype=np.int8)
    for i in range(d):
        for j in range(i + 1, d):
            if skeleton[i, j]:
                _undir(g, i, j)
    # v-structures: triple x-y-z where x->y<-z (y is collider) and x,z not adjacent
    for x in range(d):
        for y in range(d):
            if x == y:
                continue
            if adj[x, y] != 1:  # x must point to y
                continue
            for z in range(x + 1, d):
                if z == y or adj[z, y] != 1:  # z must point to y
                    continue
                if adj[x, z] == 0 and adj[z, x] == 0:  # x,z not adjacent
                    _orient(g, x, y)
                    _orient(g, z, y)
    g = meek_rules(g)
    return CPDAG(d, g, name="cpdag")


# ---------- SHD (correct unordered-pair accounting) ----------

def _pair_status(g: np.ndarray, i: int, j: int) -> str:
    a, b = int(g[i, j]), int(g[j, i])
    if a == NONE and b == NONE:
        return "none"
    if a == CIRCLE or b == CIRCLE:
        return "undir"  # PAG vs CPDAG compared as undirected skeleton
    if a == ARROW and b == TAIL:
        return "i->j"
    if a == TAIL and b == ARROW:
        return "j->i"
    if a == TAIL and b == TAIL:
        return "undir"
    return "none"


def shd_cpdag(true_g: np.ndarray, est_g: np.ndarray) -> int:
    """Structural Hamming Distance on CPDAGs.

    Each unordered pair is classified into {none, undir, i->j, j->i} and any
    mismatch scores +1. This correctly counts missing/extra/reversed/partially
    oriented edges (NOT the naive abs(adj diff), which double-counts reversals).
    """
    d = true_g.shape[0]
    s = 0
    for i in range(d):
        for j in range(i + 1, d):
            if _pair_status(true_g, i, j) != _pair_status(est_g, i, j):
                s += 1
    return s


def shd_dag(true_adj: np.ndarray, est_adj: np.ndarray) -> int:
    """SHD on DAG adjacency matrices (only missing/extra/reversed)."""
    d = true_adj.shape[0]
    s = 0
    for i in range(d):
        for j in range(i + 1, d):
            t = (true_adj[i, j] == 1) or (true_adj[j, i] == 1)
            e = (est_adj[i, j] == 1) or (est_adj[j, i] == 1)
            if t != e:
                s += 1
    return s


def _directed_pairs(g: np.ndarray) -> dict:
    d = g.shape[0]
    out = {}
    for i in range(d):
        for j in range(i + 1, d):
            if _pair_status(g, i, j) in ("i->j", "j->i"):
                out[(i, j)] = _pair_status(g, i, j)
    return out


def skeleton_f1(true_g: np.ndarray, est_g: np.ndarray) -> float:
    d = true_g.shape[0]
    tp = fp = fn = 0
    for i in range(d):
        for j in range(i + 1, d):
            t = abs(int(true_g[i, j])) > 0
            e = abs(int(est_g[i, j])) > 0
            tp += int(t and e)
            fp += int((not t) and e)
            fn += int(t and (not e))
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def orientation_scores(true_g: np.ndarray, est_g: np.ndarray):
    true_d = _directed_pairs(true_g)
    est_d = _directed_pairs(est_g)
    tp = sum(1 for k in est_d if true_d.get(k) == est_d[k])
    p = tp / len(est_d) if est_d else 0.0
    r = tp / len(true_d) if true_d else 0.0
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return p, r, f1


# ---------- BIC (decomposable) ----------

def _local_bic(X: np.ndarray, i: int, parents: list) -> float:
    n = X.shape[0]
    y = X[:, i]
    if not parents:
        rss = float(np.sum((y - y.mean()) ** 2))
        return n * np.log(rss / n + 1e-10)
    Xp = X[:, parents]
    # OLS via lstsq
    beta, *_ = np.linalg.lstsq(Xp, y, rcond=None)
    resid = y - Xp @ beta
    rss = float(np.sum(resid ** 2))
    return n * np.log(rss / n + 1e-10) + len(parents) * np.log(n)


def bic_of_dag(X: np.ndarray, adj: np.ndarray) -> float:
    """Summed local Gaussian BIC over a DAG adjacency (adj[i,j]==1 iff i->j)."""
    d = adj.shape[0]
    total = 0.0
    for i in range(d):
        parents = [p for p in range(d) if adj[p, i] == 1]
        total += _local_bic(X, i, parents)
    return total


def bic_of_cpdag(X: np.ndarray, cpdag: CPDAG) -> float:
    """BIC of the DAG formed by the *directed* edges of a CPDAG (undirected
    edges are ignored for the parent set, giving a conservative baseline)."""
    d = cpdag.d
    adj = np.zeros((d, d), dtype=int)
    g = cpdag.graph
    for i in range(d):
        for j in range(d):
            if g[i, j] == ARROW:
                adj[i, j] = 1
    return bic_of_dag(X, adj)


# ---------- acyclicity ----------

def is_dag(adj: np.ndarray) -> bool:
    adj = np.asarray(adj, dtype=int)
    d = adj.shape[0]
    indeg = adj.sum(axis=0)
    # Kahn's algorithm
    from collections import deque
    q = deque([i for i in range(d) if indeg[i] == 0])
    seen = 0
    indeg = indeg.copy()
    while q:
        u = q.popleft()
        seen += 1
        for v in range(d):
            if adj[u, v] == 1:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
    return seen == d


def power_iteration_acyclic(W: np.ndarray, max_iter: int | None = None) -> bool:
    """Independent cross-check of acyclicity: |W| has spectral radius 0 for a DAG.
    Iterating v <- |W|^T v for d steps; a DAG drives v -> 0."""
    A = np.abs(np.asarray(W, dtype=float))
    d = A.shape[0]
    steps = max_iter or d
    v = np.ones(d)
    for _ in range(steps):
        v = A.T @ v
        if np.sum(v) < _TOL:
            return True
    return False


# ---------- d-separation (oracle) ----------

def d_separation(dag_adj: np.ndarray, i: int, j: int, S: tuple) -> bool:
    """Standard d-separation on a DAG: moralize over ancestors, drop S, check
    connectivity of i and j in the undirected moral graph."""
    adj = np.asarray(dag_adj, dtype=int)
    d = adj.shape[0]
    S = set(S)

    # ancestors of {i, j} U S
    anc = set([i, j]) | S
    changed = True
    while changed:
        changed = False
        for a in list(anc):
            for p in range(d):
                if adj[p, a] == 1 and p not in anc:
                    anc.add(p)
                    changed = True
    # moral graph over ancestors
    mg = np.zeros((d, d), dtype=int)
    for a in anc:
        for b in anc:
            if a != b and adj[a, b] == 1:
                mg[a, b] = 1
                mg[b, a] = 1
    # marry parents
    for a in anc:
        pars = [p for p in range(d) if adj[p, a] == 1 and p in anc]
        for x in range(len(pars)):
            for y in range(x + 1, len(pars)):
                mg[pars[x], pars[y]] = 1
                mg[pars[y], pars[x]] = 1
    # drop S
    keep = [a for a in anc if a not in S]
    # BFS connectivity from i to j among keep
    if i not in keep or j not in keep:
        return True  # separated (one of them is in S)
    from collections import deque
    visited = {i}
    q = deque([i])
    while q:
        u = q.popleft()
        for v in keep:
            if v not in visited and mg[u, v] == 1:
                visited.add(v)
                q.append(v)
    return j not in visited
