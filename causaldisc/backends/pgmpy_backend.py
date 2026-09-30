"""SOTA backend: pgmpy (second-opinion score-based / constraint-based)."""
from __future__ import annotations

import numpy as np

from causaldisc.core.types import CPDAG, ARROW, TAIL


def available() -> bool:
    try:
        import pgmpy  # noqa: F401
        return True
    except Exception:
        return False


def _dag_to_cpdag(X, model_edges, name):
    d = X.shape[1]
    g = np.zeros((d, d), dtype=np.int8)
    for u, v in model_edges:
        i, j = int(str(u).split("x")[-1]) if "x" in str(u) else int(u), \
               int(str(v).split("x")[-1]) if "x" in str(v) else int(v)
        g[i, j] = ARROW
        g[j, i] = TAIL
    return CPDAG(d, g, name=name)


def hc(X):
    from pgmpy.estimators import HillClimbSearch
    import pandas as pd
    d = X.shape[1]
    cols = [f"x{i}" for i in range(d)]
    df = pd.DataFrame(X, columns=cols)
    # pgmpy >=1.1 renamed BicScore -> structure_score.BIC; 'bic-g' is the
    # Gaussian BIC score string (continuous data). show_progress silences tqdm.
    model = HillClimbSearch(df).estimate(scoring_method="bic-g", show_progress=False)
    return _dag_to_cpdag(X, model.edges(), "pgmpy_hc")


def pc(X, alpha=0.01):
    from pgmpy.estimators import PC
    import pandas as pd
    d = X.shape[1]
    cols = [f"x{i}" for i in range(d)]
    df = pd.DataFrame(X, columns=cols)
    model = PC(df).estimate()
    return _dag_to_cpdag(X, model.edges(), "pgmpy_pc")
