import random
import sys
from pathlib import Path

from functional_dag_compile import compile_functional_dag
from resolution import ClauseFeasibility, compile_constraints, to_dsl
from signed_clauses import parse_constraints


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
CONSTRAINTS = Path(__file__).resolve().parents[1] / "constraints" / "mnist"
sys.path.insert(0, str(EXAMPLES))

from generate_mnist_sum_digit_tree import (  # noqa: E402
    generate_mnist_sum_digit_tree_dsl,
)


def infer_sum(n_digits: int, digits: list[int]) -> int:
    constraints = parse_constraints(generate_mnist_sum_digit_tree_dsl(n_digits))
    compiled = compile_functional_dag(constraints)
    feasibility = ClauseFeasibility(compiled)
    assignment = {i: value for i, value in enumerate(digits)}
    for step, var in enumerate(compiled.ordering):
        if var in assignment:
            continue
        feasible = feasibility.feasible_set(step, assignment)
        assert len(feasible) == 1
        assignment[var] = next(iter(feasible))
    final_digits = [
        assignment[compiled.var_names.index(f"sum_{i}")]
        for i in range(n_digits.bit_length())
    ]
    value = 0
    for digit in final_digits:
        value = 10 * value + digit
    return value


def test_n2_fast_compile_matches_resolution_compile():
    constraints = parse_constraints(generate_mnist_sum_digit_tree_dsl(2))
    assert to_dsl(compile_functional_dag(constraints)) == to_dsl(
        compile_constraints(constraints)
    )


def test_n16_random_arithmetic():
    rng = random.Random(13)
    for _ in range(20):
        digits = [rng.randrange(10) for _ in range(16)]
        assert infer_sum(16, digits) == sum(digits)
