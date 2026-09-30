import numpy as np

from causaldisc.data.synthetic import load_all, chain_dgp, fork_dgp, collider_dgp, tree_dgp, dense_dgp, hard_dgp
from causaldisc.graph.primitives import shd_cpdag, skeleton_f1
from causaldisc.discover.pc import pc_stable


def test_all_datasets_shapes():
    for ds in load_all():
        assert ds["X"].shape == (ds["n"], ds["d"])
        assert ds["cpdag"].d == ds["d"]
        assert ds["n_edges"] > 0


def test_true_cpdag_edges():
    ds = chain_dgp(d=10, n=500, seed=1)
    assert ds["n_edges"] == 9
    assert ds["cpdag"].n_edges() == 9


def test_chain_recoverable():
    ds = chain_dgp(d=8, n=800, seed=1)
    est = pc_stable(ds["X"])
    assert shd_cpdag(ds["cpdag"].graph, est.graph) <= 3


def test_fork_recoverable():
    ds = fork_dgp(n=800, seed=1)
    est = pc_stable(ds["X"])
    assert skeleton_f1(ds["cpdag"].graph, est.graph) >= 0.9


def test_collider_orientation():
    ds = collider_dgp(n=1200, seed=1)
    est = pc_stable(ds["X"])
    assert shd_cpdag(ds["cpdag"].graph, est.graph) <= ds["n_edges"]


def test_tree_recoverable():
    ds = tree_dgp(d=20, n=500, seed=1)
    est = pc_stable(ds["X"])
    assert skeleton_f1(ds["cpdag"].graph, est.graph) >= 0.6


def test_dense_non_trivial():
    ds = dense_dgp(n=1000, seed=1)
    est = pc_stable(ds["X"])
    # weak edges => non-trivial (not fully broken, not perfect)
    shd = shd_cpdag(ds["cpdag"].graph, est.graph)
    assert 0 < shd < ds["n_edges"] * 2


def test_hard_runs():
    ds = hard_dgp(n=1000, seed=1)
    est = pc_stable(ds["X"])
    assert est.d == ds["d"]
