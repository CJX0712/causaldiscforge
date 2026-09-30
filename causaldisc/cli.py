"""Command-line entry point for CausalDiscForge."""
from __future__ import annotations

import argparse
import sys

import numpy as np

from causaldisc.core.config import Config
from causaldisc.pipeline.pipeline import CausalDiscPipeline


def main(argv=None):
    ap = argparse.ArgumentParser(prog="causaldisc", description="CausalDiscForge — causal discovery")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list-models", help="list available discoverers")
    pb = sub.add_parser("benchmark", help="run full benchmark, dump benchmark.json")
    pb.add_argument("--out", default="benchmark.json")
    pb.add_argument("--methods", nargs="*", default=None)
    pr = sub.add_parser("run", help="discover structure of an .npy/.csv dataset")
    pr.add_argument("--data", required=True)
    pr.add_argument("--methods", nargs="*", default=None)
    args = ap.parse_args(argv)

    pipe = CausalDiscPipeline()
    if args.cmd == "list-models" or args.cmd is None:
        print("Available discoverers:")
        for m in pipe.list_models():
            print(f"  - {m}")
    elif args.cmd == "benchmark":
        pipe.run_demo(out_path=args.out, methods=args.methods)
    elif args.cmd == "run":
        from causaldisc.eval.metrics import evaluate
        from causaldisc.data.synthetic import load_all
        X = np.load(args.data) if args.data.endswith(".npy") else np.loadtxt(args.data, delimiter=",")
        out = pipe.run(X, methods=args.methods)
        for name, cp in out.items():
            if cp is None:
                print(f"{name}: FAILED")
            else:
                print(f"{name}: {cp.n_edges()} edges")
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
