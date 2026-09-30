"""SOTA backend: lingam (non-Gaussian / ICA family — the 4th induction-bias family)."""
from __future__ import annotations

import numpy as np

from causaldisc.core.types import CPDAG, ARROW, TAIL


def available() -> bool:
    try:
        from lingam import DirectLiNGAM  # noqa: F401
        return True
    except Exception:
        return False


def direct_lingam(X):
    from lingam import DirectLiNGAM
    d = X.shape[1]
    m = DirectLiNGAM()
    m.fit(X)
    A = np.asarray(m.adjacency_matrix_, dtype=float)
    # LiNGAM: x_i = sum_j A[i,j] x_j + e_i  ->  causal edge j -> i where A[i,j] != 0
    g = np.zeros((d, d), dtype=np.int8)
    for i in range(d):
        for j in range(d):
            if i != j and abs(A[i, j]) > 1e-6:
                g[j, i] = ARROW
                g[i, j] = TAIL
    return CPDAG(d, g, name="lingam")
