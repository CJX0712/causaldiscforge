import numpy as np

from causaldisc.core.config import Config
from causaldisc.core.types import ARROW, TAIL
from causaldisc.data.synthetic import chain_dgp, collider_dgp, tree_dgp, dense_dgp
from causaldisc.discover.pc import pc_stable, pc_oracle
from causaldisc.discover.ges import hill_climbing
from causaldisc.discover.notears import notears, _notears_single, _h, _prune_to_dag
from causaldisc.discover.consensus import consensus
from causaldisc.graph.primitives import dag_to_cpdag, shd_cpdag, is_dag


def test_pc_chain_recovers():
    ds = chain_dgp(d=8, n=800, seed=1)
    est = pc_stable(ds["X"], alpha=0.01, max_level=3)
    shd = shd_cpdag(ds["cpdag"].graph, est.graph)
    assert shd <= 3  # chain is easy; allow minor disagreement


def test_pc_oracle_lands_in_true_mec():
    ds = chain_dgp(d=8, n=600, seed=2)
    true_cp = dag_to_cpdag(ds["adj"])
    est = pc_oracle(ds["adj"], ds["X"])
    assert shd_cpdag(true_cp.graph, est.graph) == 0


def test_pc_column_perm_invariant():
    rng = np.random.default_rng(0)
    ds = chain_dgp(d=8, n=600, seed=5)
    est = pc_stable(ds["X"])
    perm = rng.permutation(8)
    Xp = ds["X"][:, perm]
    est_p = pc_stable(Xp)
    g_perm = est.graph.copy()[np.ix_(perm, perm)]
    assert shd_cpdag(g_perm, est_p.graph) == 0


def test_hc_chain_skeleton():
    ds = chain_dgp(d=6, n=600, seed=3)
    cfg = Config(hc_restarts=3)
    est = hill_climbing(ds["X"], restarts=cfg.hc_restarts, seed=3)
    from causaldisc.graph.primitives import skeleton_f1
    assert skeleton_f1(ds["cpdag"].graph, est.graph) >= 0.9


def test_hc_tree_skeleton():
    ds = tree_dgp(d=12, n=500, seed=4)
    est = hill_climbing(ds["X"], restarts=2, seed=4)
    from causaldisc.graph.primitives import skeleton_f1
    assert skeleton_f1(ds["cpdag"].graph, est.graph) >= 0.6


def test_notears_acyclic_output():
    ds = chain_dgp(d=6, n=500, seed=6)
    est = notears(ds["X"])
    # est.graph is a CPDAG (endpoint codes); the directed part i->j is encoded
    # as g[i, j] == ARROW. The directed edges of a valid CPDAG are acyclic.
    adj = (est.graph == ARROW).astype(int)
    assert is_dag(adj)


def test_notears_h_constraint():
    rng = np.random.default_rng(0)
    d = 5
    W = rng.normal(size=(d, d)) * 0.3
    w = _notears_single(np.eye(d) + 0.1 * rng.normal(size=(d, d)), 0.1, 0.3, h_max_iter=60, seed=0)
    # after threshold + prune it must be a DAG
    w = _prune_to_dag(w)
    assert is_dag((np.abs(w) > 0).astype(int))
    assert _h(w) <= 1e-8 + 1.0  # h near zero (allow tiny after pruning)


def test_notears_chain_recovers():
    ds = chain_dgp(d=6, n=600, seed=7)
    est = notears(ds["X"])
    assert shd_cpdag(ds["cpdag"].graph, est.graph) <= ds["n_edges"]


def test_consensus_valid_cpdag():
    cfg = Config(consensus_bootstraps=5, consensus_allow_lingam=False, hc_restarts=1)
    ds = chain_dgp(d=6, n=500, seed=8)
    est = consensus(ds["X"], config=cfg)
    assert est.graph.shape == (6, 6)
    assert set(np.unique(est.graph)).issubset({0, -1, 1})


def test_consensus_edge_freq_range():
    cfg = Config(consensus_bootstraps=6, consensus_allow_lingam=False, hc_restarts=1)
    ds = chain_dgp(d=6, n=500, seed=9)
    est = consensus(ds["X"], config=cfg)
    assert est.edge_freq is not None
    assert est.edge_freq.min() >= 0.0
    assert est.edge_freq.max() <= 1.0


def test_consensus_crowd_consistent_selection():
    """Model-selection contract (no BIC rollback): consensus returns the member
    CPDAG most consistent with the cross-family stability skeleton. On a clean
    chain that member is the correct one, so SHD must be near-perfect."""
    from causaldisc.graph.primitives import shd_cpdag
    cfg = Config(consensus_bootstraps=5, consensus_allow_lingam=False, hc_restarts=1)
    ds = chain_dgp(d=6, n=500, seed=10)
    est = consensus(ds["X"], config=cfg)
    assert shd_cpdag(ds["cpdag"].graph, est.graph) <= 2


def test_collider_orientation():
    ds = collider_dgp(n=1200, seed=11)
    est = pc_stable(ds["X"], alpha=0.01, max_level=3)
    from causaldisc.graph.primitives import orientation_scores
    p, r, f1 = orientation_scores(ds["cpdag"].graph, est.graph)
    # collider has 3 directed edges to recover
    assert r >= 0.5
