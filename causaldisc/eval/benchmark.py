"""Benchmark harness: run a method registry across synthetic datasets."""
from __future__ import annotations

import time

import numpy as np

from causaldisc.core.config import Config
from causaldisc.discover.pc import pc_stable
from causaldisc.discover.ges import hill_climbing
from causaldisc.discover.notears import notears
from causaldisc.discover.consensus import consensus
from causaldisc.backends.registry import available_backends
from causaldisc.eval.metrics import evaluate


def build_method_registry(config: Config | None = None):
    config = config or Config()
    reg = {
        "pc": lambda X: pc_stable(X, alpha=config.pc_alpha, max_level=config.pc_max_level),
        "hc": lambda X: hill_climbing(X, restarts=config.hc_restarts,
                                      max_parents=config.hc_max_parents, seed=config.random_seed),
        "notears": lambda X: notears(X, lambdas=config.notears_lambdas,
                                     w_thresholds=config.notears_w_thresholds,
                                     h_max_iter=100, seed=config.random_seed),
        "consensus": lambda X: consensus(X, config=config),
    }
    # SOTA backends (only those available -> never fake a number)
    for name, fn in available_backends().items():
        reg[name] = fn
    return reg


def run_benchmark(datasets, config: Config | None = None, methods=None, verbose=False):
    config = config or Config()
    reg = build_method_registry(config)
    if methods:
        reg = {k: reg[k] for k in methods if k in reg}
    rows = []
    for ds in datasets:
        X = ds["X"]
        true_cp = ds["cpdag"]
        for name, fn in reg.items():
            t0 = time.time()
            try:
                est = fn(X)
            except Exception as e:
                if verbose:
                    print(f"  ! {name} failed on {ds['name']}: {e}")
                continue
            dt = time.time() - t0
            metrics = evaluate(true_cp, est, X)
            metrics.update({"dataset": ds["name"], "method": name, "runtime_s": round(dt, 3)})
            rows.append(metrics)
    return rows


def print_table(rows):
    header = ["dataset", "method", "shd", "skel_f1", "or_f1", "n_est", "runtime"]
    print(f"{'dataset':<10}{'method':<12}{'shd':>5}{'skelF1':>8}{'orF1':>7}{'nEst':>6}{'time':>8}")
    print("-" * 56)
    for r in rows:
        print(f"{r['dataset']:<10}{r['method']:<12}{r['shd_cpdag']:>5}"
              f"{r['skeleton_f1']:>8}{r['orient_f1']:>7}{r['n_est_edges']:>6}{r['runtime_s']:>8}")
