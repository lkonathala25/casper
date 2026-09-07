import sys
from pathlib import Path

from resolution import enumerate_models
from signed_clauses import parse_constraints


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
CONSTRAINTS = Path(__file__).resolve().parents[1] / "constraints" / "mnist"
sys.path.insert(0, str(EXAMPLES))

from generate_mnist_sum_tree import generate_mnist_sum_tree_dsl  # noqa: E402


def test_n2_tree_models_are_exact_sums():
    constraints = parse_constraints(generate_mnist_sum_tree_dsl(2))
    models = enumerate_models(constraints.clauses, constraints.var_domains)
    assert len(models) == 100
    assert all(y1 + y2 == total for y1, y2, total in models)


def test_n16_tree_shape():
    constraints = parse_constraints(generate_mnist_sum_tree_dsl(16))
    assert len(constraints.var_names) == 31
    assert constraints.var_domains[-1] == 145
    assert len(constraints.clauses) == 10_311
