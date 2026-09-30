"""High-level pipeline: run a method registry, run benchmarks."""
from __future__ import annotations

import json
import os

import numpy as np

from causaldisc.core.config import Config
from causaldisc.eval.benchmark import build_method_registry, run_benchmark
from causaldisc.data.synthetic import load_all


class CausalDiscPipeline:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()

    def list_models(self):
        return list(build_method_registry(self.config).keys())

    def run(self, X, methods=None):
        reg = build_method_registry(self.config)
        if methods:
            reg = {k: reg[k] for k in methods if k in reg}
        out = {}
        for name, fn in reg.items():
            try:
                out[name] = fn(np.asarray(X, float))
            except Exception as e:
                out[name] = None
        return out

    def benchmark(self, datasets=None, methods=None):
        if datasets is None:
            datasets = load_all()
        return run_benchmark(datasets, self.config, methods=methods)

    def run_demo(self, out_path="benchmark.json", methods=None, verbose=True):
        datasets = load_all()
        rows = self.benchmark(datasets=datasets, methods=methods)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        if verbose:
            from causaldisc.eval.benchmark import print_table
            print_table(rows)
            self._print_aggregate(rows)
        return rows

    @staticmethod
    def _print_aggregate(rows):
        agg = {}
        for r in rows:
            agg.setdefault(r["method"], []).append(r["shd_cpdag"])
        print("\nAggregate SHD by method (mean over datasets):")
        for m, vals in sorted(agg.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
            print(f"  {m:<12} mean={sum(vals)/len(vals):.2f}  per={vals}")
