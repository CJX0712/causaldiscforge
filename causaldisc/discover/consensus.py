"""ConsensusCD — flagship ensemble (cross-induction-bias model selection).

Strategy (robust, truth-free):
  - Bootstrap the data B times; for each resample run every member algorithm and
    tally, per edge, (a) how often it appears (``skel_freq`` -> edge confidence
    ``edge_freq`` in [0,1]) and (b) which *algorithm families* confirmed it.
  - Stability skeleton ``S*`` = an edge is kept only if it is frequent enough
    (``skel_freq >= tau``) AND confirmed by >= 2 *distinct* families
    (constraint / score / continuous / ica). This cross-induction-bias test
    filters out spurious edges that a single (noisy) method would otherwise
    inject, which is what makes naive edge-voting ensembles worse than the best
    single method.
  - Run every member on the FULL data to get candidate CPDAGs, then *select* the
    candidate whose skeleton best agrees with S* (the most "crowd-consistent"
    model). This is a model-selection ensemble: it inherits the accuracy of the
    crowd-consistent member and is robust to any single noisy method. When the
    stability skeleton is reliable it matches or beats any single base method;
    when every method disagrees with S* the ensemble degrades gracefully to the
    most-supported candidate rather than inventing structure.
  - NOTE: we deliberately do NOT roll back to the lowest-BIC base member. A BIC
    tie-break is fine as a *last-resort* tiebreak, but using BIC as an
    out-of-band override was empirically found to *increase* CPDAG error because
    BIC favours denser/degenerate graphs that score worse on the true CPDAG.

Members span 3 core families (constraint PC, score Hill-Climbing/BIC, continuous
NOTEARS) plus optional stronger SOTA backends (causal-learn PC/GES) and optional
LiNGAM (off by default — LiNGAM targets non-Gaussian data and is a poor member on
linear-Gaussian benchmarks, where it pollutes the stability votes).
"""
from __future__ import annotations

import numpy as np

from causaldisc.core.config import Config
from causaldisc.core.types import CPDAG, NONE, TAIL, ARROW, CIRCLE
from causaldisc.graph.primitives import bic_of_cpdag
from causaldisc.discover.pc import pc_stable
from causaldisc.discover.ges import hill_climbing
from causaldisc.discover.notears import notears
from causaldisc.backends import lingam_backend


# core algorithm family each member belongs to (used for cross-bias confirmation)
_FAMILY = {
    "pc": "constraint", "cl_pc": "constraint",
    "hc": "score", "cl_ges": "score",
    "nt": "continuous",
    "lingam": "ica",
}


def _normalize_endpoints(g: np.ndarray) -> np.ndarray:
    """Coerce any CIRCLE (PAG) endpoint to TAIL so the result is a valid CPDAG."""
    g = g.copy()
    g[g == CIRCLE] = TAIL
    return g


def consensus(X, config: Config | None = None, seed: int | None = None) -> CPDAG:
    config = config or Config()
    if seed is None:
        seed = config.random_seed
    X = np.asarray(X, float)
    n, d = X.shape

    # ---- member registry (3 families + optional stronger SOTA backends) ----
    members = [
        ("pc", lambda Xx: pc_stable(Xx, alpha=config.pc_alpha, max_level=config.pc_max_level)),
        ("hc", lambda Xx: hill_climbing(Xx, restarts=config.hc_restarts,
                                       max_parents=config.hc_max_parents, seed=seed)),
        ("nt", lambda Xx: notears(Xx, lambdas=config.notears_lambdas,
                                  w_thresholds=config.notears_w_thresholds,
                                  h_max_iter=100, seed=seed)),
    ]
    try:  # stronger SOTA backends, auto-detected
        from causaldisc.backends.causallearn_backend import pc as cl_pc, ges as cl_ges
        members.append(("cl_pc", lambda Xx: cl_pc(Xx)))
        members.append(("cl_ges", lambda Xx: cl_ges(Xx)))
    except Exception:
        pass
    if config.consensus_allow_lingam and lingam_backend.available():
        members.append(("lingam", lambda Xx: lingam_backend.direct_lingam(Xx)))

    # ---- bootstrap stability: edge frequency + cross-family confirmation ----
    skel_freq = np.zeros((d, d), dtype=float)
    fam_presence = {}  # (i,j) -> set of families that confirmed it (across bootstraps)
    rng = np.random.default_rng(seed)
    for _ in range(config.consensus_bootstraps):
        idx = rng.choice(n, size=n // 2, replace=False)
        Xb = X[idx]
        fam_seen = {}  # (i,j) -> set of families seen this bootstrap
        for name, fn in members:
            try:
                cpdag = fn(Xb)
            except Exception:
                continue
            fam = _FAMILY.get(name, "other")
            for (i, j) in cpdag.edge_list():
                ii, jj = (i, j) if i < j else (j, i)
                fam_seen.setdefault((ii, jj), set()).add(fam)
        for (e, fams) in fam_seen.items():
            skel_freq[e] += 1
            fam_presence[e] = fam_presence.get(e, set()) | fams

    skel_frac = skel_freq / config.consensus_bootstraps            # in [0, 1]
    # stability skeleton: frequent AND confirmed by >= 2 distinct families
    S_star = (skel_frac >= config.consensus_tau)
    for (i, j), fams in fam_presence.items():
        if len(fams) < 2:
            S_star[i, j] = False

    # ---- full-data candidate CPDAGs ----
    candidates = {}
    for name, fn in members:
        try:
            candidates[name] = fn(X)
        except Exception:
            continue

    if not candidates:
        g = np.zeros((d, d), dtype=np.int8)
        return CPDAG(d, g, edge_freq=skel_frac, name="consensus")

    # ---- stability-guided model selection ----
    def _skel_shd_to_S(cp: CPDAG) -> int:
        s = 0
        for i in range(d):
            for j in range(i + 1, d):
                t = bool(S_star[i, j])
                e = abs(int(cp.graph[i, j])) > 0 or abs(int(cp.graph[j, i])) > 0
                if t != e:
                    s += 1
        return s

    def _num_directed(cp: CPDAG) -> int:
        return int(np.sum(cp.graph == ARROW))

    scored = []
    for name, cp in candidates.items():
        # Selection is pure model-selection among the member CPDAGs, driven by
        # agreement with the cross-family stability skeleton S* (truth-free):
        #   1) skeleton best matches S*  (the crowd-consistent structure)
        #   2) among ties, prefer the simplest (fewest oriented edges) — for
        #      chain/fork truths the undirected method (PC) is the correct one,
        #      and a raw BIC tie-break is misled into preferring needless
        #      orientation (BIC is anti-correlated with SHD on those graphs)
        #   3) final tie-break: lowest BIC
        # We deliberately do NOT roll back to the lowest-BIC base member: that
        # "non-dominance guard" was found to *increase* SHD because BIC favours
        # denser/degenerate graphs that score worse on the true CPDAG.
        scored.append((_skel_shd_to_S(cp), _num_directed(cp), bic_of_cpdag(X, cp), name, cp))
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    _, _, _, best_name, best_cp = scored[0]

    g = _normalize_endpoints(best_cp.graph)
    return CPDAG(d, g, edge_freq=skel_frac, name=f"consensus:{best_name}")
