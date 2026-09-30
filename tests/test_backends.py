import numpy as np
import pytest

from causaldisc.core.types import ARROW, TAIL
from causaldisc.graph.primitives import dag_to_cpdag, shd_cpdag, skeleton_f1
from causaldisc.data.synthetic import chain_dgp

causallearn = pytest.importorskip("causallearn")
pgmpy = pytest.importorskip("pgmpy")
lingam = pytest.importorskip("lingam")

from causaldisc.backends.causallearn_backend import pc as cl_pc, ges as cl_ges
from causaldisc.backends.pgmpy_backend import hc as pg_hc
from causaldisc.backends.lingam_backend import direct_lingam
from causaldisc.discover.pc import pc_oracle


def test_causallearn_pc_codes():
    ds = chain_dgp(d=6, n=800, seed=1)
    cp = cl_pc(ds["X"])
    assert set(np.unique(cp.graph)).issubset({0, -1, 1, 2})
    assert skeleton_f1(ds["cpdag"].graph, cp.graph) >= 0.9


def test_causallearn_pc_vs_oracle():
    ds = chain_dgp(d=8, n=800, seed=2)
    cp = cl_pc(ds["X"])
    oracle = pc_oracle(ds["adj"], ds["X"])
    # external cross-check (invariant 12): should be near-identical
    assert shd_cpdag(oracle.graph, cp.graph) <= 2


def test_causallearn_ges():
    ds = chain_dgp(d=6, n=800, seed=3)
    cp = cl_ges(ds["X"])
    assert skeleton_f1(ds["cpdag"].graph, cp.graph) >= 0.9


def test_pgmpy_hc():
    ds = chain_dgp(d=6, n=800, seed=4)
    cp = pg_hc(ds["X"])
    assert skeleton_f1(ds["cpdag"].graph, cp.graph) >= 0.9


def test_lingam_direct():
    ds = chain_dgp(d=5, n=800, seed=5)
    cp = direct_lingam(ds["X"])
    assert cp.d == 5
    # LiNGAM on a linear-Gaussian chain: skeleton should be well recovered
    assert skeleton_f1(ds["cpdag"].graph, cp.graph) >= 0.6
