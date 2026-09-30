"""End-to-end demo: benchmark across all synthetic datasets, print + dump JSON."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causaldisc.pipeline.pipeline import CausalDiscPipeline


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "benchmark.json")
    pipe = CausalDiscPipeline()
    rows = pipe.run_demo(out_path=out, verbose=True)
    print(f"\n[run_demo] wrote {len(rows)} result rows -> {out}")


if __name__ == "__main__":
    main()
