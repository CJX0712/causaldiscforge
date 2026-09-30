"""SOTA backend: causal-learn (CMU). PC / GES / CDNOD / FCI.

causal-learn returns matrices already coded with the same endpoint convention
we use (-1 tail, 1 arrow, 2 circle), so adaptation is a plain cast.
"""
from __future__ import annotations

import numpy as np

from causaldisc.core.types import CPDAG, ARROW, TAIL


def available() -> bool:
    try:
        import causallearn  # noqa: F401
        return True
    except Exception:
        return False


def _from_cl_graph(X, cg_graph, name, pag=False):
    d = X.shape[1]
    g = np.asarray(cg_graph, dtype=np.int8).reshape(d, d)
    return CPDAG(d, g, name=name, pag=pag)


def pc(X, alpha=0.01):
    from causallearn.search.ConstraintBased.PC import pc as cl_pc
    cg = cl_pc(X, alpha=alpha, show_progress=False)
    return _from_cl_graph(X, cg.G.graph, "cl_pc")


def ges(X):
    from causallearn.search.ScoreBased.GES import ges as cl_ges
    out = cl_ges(X)
    return _from_cl_graph(X, out["G"].graph, "cl_ges")


def cdnod(X):
    from causallearn.search.ConstraintBased.CDNOD import cdnod as cl_cdnod
    out = cl_cdnod(X, c_indx=[])  # no latent index -> standard CD
    cg = out[0]
    return _from_cl_graph(X, cg.G.graph, "cl_cdnod")


def fci(X):
    from causallearn.search.ConstraintBased.FCI import fci as cl_fci
    out = cl_fci(X)
    cg = out[0]
    return _from_cl_graph(X, cg.G.graph, "cl_fci", pag=True)
