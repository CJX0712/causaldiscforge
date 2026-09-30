import numpy as np
import pytest

from causaldisc.core.types import NONE, TAIL, ARROW
from causaldisc.graph.primitives import (
    dag_to_cpdag, shd_cpdag, partial_corr, is_dag, power_iteration_acyclic,
    d_separation, bic_of_dag, _local_bic, skeleton_f1, orientation_scores,
)
from causaldisc.data.synthetic import chain_dgp, collider_dgp


def _chain_adj(d=6):
    adj = np.zeros((d, d), int)
    for i in range(d - 1):
        adj[i, i + 1] = 1
    return adj


def test_dag_to_cpdag_chain_all_undirected():
    adj = _chain_adj(5)
    cp = dag_to_cpdag(adj)
    # a chain has no v-structures -> every present edge is undirected. Check the
    # off-diagonal edge entries are TAIL (non-edges stay NONE by convention).
    off = ~np.eye(5, dtype=bool)
    edge_mask = cp.graph[off] != NONE
    assert np.all(cp.graph[off][edge_mask] == TAIL)


def test_dag_to_cpdag_collider_orients():
    adj = np.zeros((3, 3), int)
    adj[0, 2] = 1
    adj[1, 2] = 1  # 0->2<-1
    cp = dag_to_cpdag(adj)
    assert cp.graph[0, 2] == ARROW and cp.graph[2, 0] == TAIL
    assert cp.graph[1, 2] == ARROW and cp.graph[2, 1] == TAIL


def test_dag_to_cpdag_diamond():
    adj = np.zeros((4, 4), int)
    adj[0, 1] = 1
    adj[0, 2] = 1
    adj[1, 3] = 1
    adj[2, 3] = 1  # 1->3<-2 collider; 0->1, 0->2
    cp = dag_to_cpdag(adj)
    assert cp.graph[1, 3] == ARROW and cp.graph[2, 3] == ARROW


def test_mec_invariance():
    adj_a = _chain_adj(5)
    # a different DAG in the same MEC (reverse some edges, keep skeleton undirected)
    adj_b = np.zeros((5, 5), int)
    adj_b[1, 0] = 1
    adj_b[1, 2] = 1
    adj_b[2, 3] = 1
    adj_b[3, 4] = 1
    cp_a = dag_to_cpdag(adj_a)
    cp_b = dag_to_cpdag(adj_b)
    assert shd_cpdag(cp_a.graph, cp_b.graph) == 0


def test_shd_empty():
    adj = _chain_adj(5)
    cp = dag_to_cpdag(adj)
    empty = np.zeros((5, 5), int)
    assert shd_cpdag(empty, cp.graph) == cp.n_edges()


def test_shd_perfect():
    adj = _chain_adj(5)
    cp = dag_to_cpdag(adj)
    assert shd_cpdag(cp.graph, cp.graph) == 0


def test_partial_corr_independent():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 3))
    # fork: x2 is a common cause of x0 and x1  ->  0 and 1 independent given 2
    X[:, 2] = rng.normal(0, 1, 1000)
    X[:, 0] = X[:, 2] + rng.normal(0, 0.3, 1000)
    X[:, 1] = X[:, 2] + rng.normal(0, 0.3, 1000)
    r = partial_corr(X, 0, 1, (2,))
    assert abs(r) < 0.05


def test_partial_corr_perfect():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1000, 2))
    r = partial_corr(X, 0, 1, ())
    assert abs(abs(r) - abs(np.corrcoef(X.T)[0, 1])) < 1e-6


def test_is_dag():
    assert is_dag(_chain_adj(5))
    cyc = np.zeros((3, 3), int)
    cyc[0, 1] = cyc[1, 2] = cyc[2, 0] = 1
    assert not is_dag(cyc)


def test_power_iteration_acyclic_dag():
    W = np.zeros((6, 6))
    W[0, 1] = 0.5
    W[1, 2] = 0.3
    assert power_iteration_acyclic(W)


def test_power_iteration_acyclic_cycle():
    W = np.zeros((3, 3))
    W[0, 1] = W[1, 2] = W[2, 0] = 1.0
    assert not power_iteration_acyclic(W)


def test_d_separation_chain():
    adj = _chain_adj(4)
    assert d_separation(adj, 0, 2, (1,))
    assert not d_separation(adj, 0, 2, ())


def test_d_separation_block():
    adj = _chain_adj(4)
    assert d_separation(adj, 0, 3, (1, 2))
    # conditioning on the middle node of the chain 0->1->2->3 blocks the path
    assert d_separation(adj, 0, 3, (1,))


def test_bic_decomposable_incremental():
    X = chain_dgp(d=6, n=400, seed=3)["X"]
    adj0 = np.zeros((6, 6), int)
    adj1 = adj0.copy()
    adj1[0, 1] = 1
    delta_global = bic_of_dag(X, adj1) - bic_of_dag(X, adj0)
    delta_local = _local_bic(X, 1, [0]) - _local_bic(X, 1, [])
    assert abs(delta_global - delta_local) < 1e-6


def test_skeleton_f1():
    d = 4
    true = np.zeros((d, d), int)
    true[0, 1] = true[1, 2] = TAIL
    est = np.zeros((d, d), int)
    est[0, 1] = TAIL
    est[2, 3] = TAIL  # one TP, one FP, one FN
    f1 = skeleton_f1(true, est)
    assert 0.0 < f1 < 1.0


def test_orientation_scores():
    d = 3
    true = np.zeros((d, d), int)
    true[0, 1] = ARROW
    true[1, 0] = TAIL  # 0->1
    est = true.copy()
    p, r, f1 = orientation_scores(true, est)
    assert p == 1.0 and r == 1.0 and f1 == 1.0
