#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "casper", Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from functional_dag_compile import compile_functional_dag  # noqa: E402
from generate_mnist_sum_digit_tree import (  # noqa: E402
    generate_mnist_sum_digit_tree_dsl,
)
from resolution import to_dsl  # noqa: E402
from signed_clauses import parse_constraints  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", nargs="+", type=int, default=[2, 16, 64, 1024])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark_results" / "mnist_sum_digit_tree",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n_digits in args.n:
        print(f"Compiling N={n_digits}...", flush=True)
        start = time.perf_counter()
        dsl = generate_mnist_sum_digit_tree_dsl(n_digits)
        generation_s = time.perf_counter() - start

        start = time.perf_counter()
        constraints = parse_constraints(dsl)
        parse_s = time.perf_counter() - start

        start = time.perf_counter()
        compiled = compile_functional_dag(constraints)
        compile_s = time.perf_counter() - start

        output = args.output_dir / f"mnist_sum_digit_tree_n{n_digits}_compiled.txt"
        start = time.perf_counter()
        output.write_text(to_dsl(compiled))
        write_s = time.perf_counter() - start
        row = {
            "n_digits": n_digits,
            "variables": len(constraints.var_names),
            "clauses": len(constraints.clauses),
            "generation_s": generation_s,
            "parse_s": parse_s,
            "compile_s": compile_s,
            "artifact_write_s": write_s,
            "total_pipeline_s": generation_s + parse_s + compile_s + write_s,
            "artifact_mb": output.stat().st_size / (1024 * 1024),
        }
        rows.append(row)
        print(
            f"  clauses={row['clauses']}, compile={compile_s:.3f}s, "
            f"pipeline={row['total_pipeline_s']:.3f}s",
            flush=True,
        )

    csv_path = args.output_dir / "compilation_times.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
