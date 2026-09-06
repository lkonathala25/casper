#!/usr/bin/env python3
"""Compile and benchmark MNIST-Add constraints for several operand lengths.

Compilation is deliberately serial: the functional MNIST-Add encoding keeps
no resolvents, so repeatedly creating a process pool is counterproductive.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from carry_chain_constraint_layer import (  # noqa: E402
    CarryChainConstraintLayer,
    compute_carry_chain,
)
from compiled_layer import CompiledConstraintLayer  # noqa: E402
from functional_compiled_layer import FunctionalCompiledConstraintLayer  # noqa: E402
from generate_mnist_add import generate_mnist_add_dsl  # noqa: E402
from resolution import compile_constraints, to_dsl  # noqa: E402
from signed_clauses import parse_constraints  # noqa: E402

import sys
from pathlib import Path as _P
_ARCHIVE = _P(__file__).resolve().parents[1] / "archive" / "mnist_closed_form"
if str(_ARCHIVE) not in sys.path:
    sys.path.append(str(_ARCHIVE))


DEFAULT_N = (1, 2, 4, 15, 100, 256)
BACKENDS = ("generic_compiled", "functional_compiled", "hardcoded_carry")


def segmented_random_probs(
    variable_ranges: list[int], batch_size: int, generator: torch.Generator
) -> torch.Tensor:
    return torch.cat([
        torch.softmax(torch.randn(batch_size, h, generator=generator), dim=1)
        for h in variable_ranges
    ], dim=1)


def output_predictions(
    adjusted: torch.Tensor, n_digits: int
) -> torch.Tensor:
    n_inputs = 2 * n_digits
    view = adjusted.view(adjusted.shape[0], 3 * n_digits + 1, 10)
    return view[:, n_inputs:].argmax(dim=2)


def expected_outputs(probs: torch.Tensor, n_digits: int) -> torch.Tensor:
    inputs = probs[:, : 20 * n_digits].view(
        probs.shape[0], 2 * n_digits, 10
    ).argmax(dim=2)
    return compute_carry_chain(inputs[:, :n_digits], inputs[:, n_digits:])


def time_layer(
    layer,
    probs: torch.Tensor,
    expected: torch.Tensor,
    warmups: int,
    repetitions: int,
) -> dict:
    with torch.no_grad():
        for _ in range(warmups):
            layer(probs)

        samples = []
        adjusted = None
        for _ in range(repetitions):
            start = time.perf_counter()
            adjusted, _ = layer(probs)
            samples.append(time.perf_counter() - start)

    predicted = output_predictions(adjusted, (adjusted.shape[1] // 10 - 1) // 3)
    exact = (predicted == expected).all(dim=1).float().mean().item()
    digit = (predicted == expected).float().mean().item()
    return {
        "inference_mean_s": statistics.fmean(samples),
        "inference_std_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "inference_ms_per_sample": 1000 * statistics.fmean(samples) / probs.shape[0],
        "exact_accuracy_pct": 100 * exact,
        "digit_accuracy_pct": 100 * digit,
    }


def benchmark_one(
    n_digits: int,
    output_dir: Path,
    batch_size: int,
    warmups: int,
    repetitions: int,
    seed: int,
) -> list[dict]:
    # Make the serial choice explicit even if the caller exported SAT_WORKERS.
    os.environ["SAT_WORKERS"] = "1"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / f"mnist_add_n{n_digits}.txt"
    compiled_path = output_dir / f"mnist_add_n{n_digits}_compiled.txt"

    start = time.perf_counter()
    dsl = generate_mnist_add_dsl(n_digits)
    generation_s = time.perf_counter() - start
    source_path.write_text(dsl)

    start = time.perf_counter()
    constraints = parse_constraints(dsl)
    parse_s = time.perf_counter() - start

    start = time.perf_counter()
    compiled = compile_constraints(constraints, verbose=False)
    compile_s = time.perf_counter() - start

    start = time.perf_counter()
    compiled_path.write_text(to_dsl(compiled))
    artifact_write_s = time.perf_counter() - start

    clause_count = len(constraints.clauses)
    compiled_clause_count = sum(len(bucket) for bucket in compiled.per_var_clauses)
    original_clauses = set(constraints.clauses)
    retained_resolvents = sum(
        clause not in original_clauses
        for bucket in compiled.per_var_clauses
        for clause in bucket
    )

    variable_ranges = [10] * (3 * n_digits + 1)
    generator = torch.Generator().manual_seed(seed + n_digits)
    probs = segmented_random_probs(variable_ranges, batch_size, generator)
    expected = expected_outputs(probs, n_digits)

    layers = {}
    load_times = {}
    for name, constructor in (
        ("generic_compiled", lambda: CompiledConstraintLayer(
            compiled_path, variable_ranges
        )),
        ("functional_compiled", lambda: FunctionalCompiledConstraintLayer(
            compiled_path, variable_ranges
        )),
        ("hardcoded_carry", lambda: CarryChainConstraintLayer(n_digits)),
    ):
        start = time.perf_counter()
        layers[name] = constructor()
        load_times[name] = time.perf_counter() - start

    common = {
        "n_digits": n_digits,
        "variables": len(constraints.var_names),
        "input_clauses": clause_count,
        "compiled_clauses": compiled_clause_count,
        "retained_resolvents": retained_resolvents,
        "workers": 1,
        "generation_s": generation_s,
        "parse_s": parse_s,
        "compile_s": compile_s,
        "artifact_write_s": artifact_write_s,
        "artifact_mb": compiled_path.stat().st_size / (1024 * 1024),
        "batch_size": batch_size,
        "warmups": warmups,
        "repetitions": repetitions,
    }
    rows = []
    for backend in BACKENDS:
        result = time_layer(
            layers[backend], probs, expected, warmups, repetitions
        )
        rows.append({
            **common,
            "backend": backend,
            "layer_load_s": load_times[backend],
            **result,
        })
    return rows


def write_markdown(rows: list[dict], path: Path) -> None:
    columns = (
        "N", "vars", "clauses", "compile s", "artifact MB", "backend",
        "load s", "infer ms/batch", "ms/sample", "exact acc %",
    )
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---:" if i not in (5,) else "---" for i in range(len(columns))) + "|",
    ]
    for row in rows:
        lines.append(
            f"| {row['n_digits']} | {row['variables']} | "
            f"{row['input_clauses']} | {row['compile_s']:.6f} | "
            f"{row['artifact_mb']:.3f} | {row['backend']} | "
            f"{row['layer_load_s']:.6f} | "
            f"{1000 * row['inference_mean_s']:.3f} | "
            f"{row['inference_ms_per_sample']:.6f} | "
            f"{row['exact_accuracy_pct']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", nargs="+", type=int, default=list(DEFAULT_N))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_results")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if any(n < 1 for n in args.n):
        parser.error("all --n values must be >= 1")
    if args.batch_size < 1 or args.repetitions < 1 or args.warmups < 0:
        parser.error("batch size/repetitions must be positive; warmups non-negative")

    rows = []
    for n_digits in args.n:
        print(f"Benchmarking N={n_digits} (serial compile)...", flush=True)
        new_rows = benchmark_one(
            n_digits, args.output_dir, args.batch_size,
            args.warmups, args.repetitions, args.seed,
        )
        rows.extend(new_rows)
        print(
            f"  compile={new_rows[0]['compile_s']:.3f}s, "
            f"clauses={new_rows[0]['input_clauses']}, "
            f"generic infer={1000 * new_rows[0]['inference_mean_s']:.3f}ms/batch",
            flush=True,
        )

    csv_path = args.output_dir / "mnist_add_compile_inference.csv"
    md_path = args.output_dir / "mnist_add_compile_inference.md"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
