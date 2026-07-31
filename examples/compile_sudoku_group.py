from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from resolution import compile_constraints, to_json
from signed_clauses import parse_constraints
from sudoku_groups import group_dsl


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compile one ALLDIFFERENT group (a clique of `size` "
        "mutually-distinct variables) for per-group Sudoku inference."
    )
    p.add_argument("--size", type=int, default=9)
    p.add_argument("--value-base", type=int, default=1)
    p.add_argument("--cap", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dsl-out", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    out = args.out or ROOT / "examples" / f"sudoku_group{args.size}_compiled.json"

    dsl = group_dsl(args.size, value_base=args.value_base)
    if args.dsl_out:
        args.dsl_out.write_text(dsl)
        print(f"wrote DSL to {args.dsl_out}", flush=True)

    cs = parse_constraints(dsl)
    print(
        f"group size={args.size} vars={len(cs.var_names)} "
        f"clauses={len(cs.clauses)} cap={args.cap}",
        flush=True,
    )

    t0 = time.time()
    cc = compile_constraints(cs, verbose=args.verbose, cap=args.cap)
    elapsed = time.time() - t0

    out.write_text(json.dumps(to_json(cc)))
    print(f"compiled in {elapsed:.2f}s", flush=True)
    print(f"per-step clause counts: {[len(x) for x in cc.per_var_clauses]}", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
