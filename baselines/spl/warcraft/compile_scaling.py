from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path


def peak_rss_gb() -> float:
    """Peak resident memory of this process so far, in GB."""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)


def compile_paths(g: int, cap: int | None) -> dict:
    """SPL's original encoding: graphillion's ZDD of every simple s-t path on the g x g grid.

    This is the first stage of data/compile_paths.py; its ZDD is what zdd2sdd
    converts into SPL's circuit, so an SDD cannot exist before it does. `cap`
    keeps paths of at most that many edges, as paths.smaller(33) does at 12x12.
    """
    from graphillion import GraphSet
    import graphillion.tutorial as tl

    GraphSet.set_universe(tl.grid(g - 1, g - 1))
    t0 = time.time()
    paths = GraphSet.paths(1, g * g)
    print(f"paths() done at {time.time() - t0:.1f} s, peak {peak_rss_gb()} GB",
          file=sys.stderr, flush=True)
    if cap is not None:
        paths = paths.smaller(cap + 1)
    elapsed = time.time() - t0
    zdd_nodes = len(paths.dumps().splitlines()) - 1
    count = paths.len()
    return {"encoding": "paths", "g": g, "cap": cap, "edges": 2 * g * (g - 1),
            "compile_s": round(elapsed, 2), "zdd_nodes": zdd_nodes,
            "models_log10": round(math.log10(count), 2) if count else None,
            "peak_rss_gb": peak_rss_gb()}


def compile_steps(g: int, out_dir: Path) -> dict:
    """The steps encoding: compile the walk SDD with steps_sdd.build."""
    from steps_sdd import build

    summary = build(g, out_dir)
    return {"encoding": "steps", **summary, "peak_rss_gb": peak_rss_gb()}


def main():
    """Compile one encoding at one grid size and print a JSON summary line."""
    p = argparse.ArgumentParser()
    p.add_argument("--encoding", choices=("paths", "steps"), required=True)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--cap", type=int, default=None,
                   help="paths only: keep paths of at most this many edges")
    p.add_argument("--out", type=Path, default=None,
                   help="steps only: where to write the SDD (default: steps_sdd's)")
    args = p.parse_args()
    if args.encoding == "paths":
        result = compile_paths(args.size, args.cap)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from steps_sdd import DEFAULT_OUT
        result = compile_steps(args.size, args.out or DEFAULT_OUT)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
