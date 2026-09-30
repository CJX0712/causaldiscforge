import json
import os
import sys

from causaldisc.core.config import Config
from causaldisc.pipeline.pipeline import CausalDiscPipeline
from causaldisc.data.synthetic import chain_dgp
from causaldisc.eval.metrics import evaluate
from causaldisc.graph.primitives import shd_cpdag


def test_pipeline_list_models():
    pipe = CausalDiscPipeline()
    models = pipe.list_models()
    assert "pc" in models and "hc" in models and "notears" in models and "consensus" in models


def test_pipeline_run():
    pipe = CausalDiscPipeline()
    ds = chain_dgp(d=6, n=500, seed=1)
    out = pipe.run(ds["X"], methods=["pc", "hc"])
    assert out["pc"] is not None and out["hc"] is not None
    assert out["pc"].d == 6


def test_evaluate_metrics():
    ds = chain_dgp(d=6, n=500, seed=2)
    est = pipe_res = CausalDiscPipeline().run(ds["X"], methods=["pc"])["pc"]
    m = evaluate(ds["cpdag"], est, ds["X"])
    assert "shd_cpdag" in m and "skeleton_f1" in m and "bic" in m
    assert 0.0 <= m["skeleton_f1"] <= 1.0


def test_benchmark_rows():
    pipe = CausalDiscPipeline(Config(consensus_bootstraps=4, consensus_allow_lingam=False,
                                     hc_restarts=1))
    rows = pipe.benchmark(methods=["pc", "hc", "notears", "consensus"])
    assert len(rows) >= 6 * 4
    for r in rows:
        assert r["method"] in {"pc", "hc", "notears", "consensus"}
        assert r["shd_cpdag"] >= 0


def test_cli_list_models(capsys):
    from causaldisc.cli import main
    main(["list-models"])
    out = capsys.readouterr().out
    assert "pc" in out and "consensus" in out


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("CAUSALDISC_BOOTSTRAPS", "7")
    cfg = Config()
    assert cfg.consensus_bootstraps == 7
