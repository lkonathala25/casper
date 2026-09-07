import sys
from pathlib import Path

from resolution import compile_constraints, to_dsl
from signed_clauses import parse_constraints


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
CONSTRAINTS = Path(__file__).resolve().parents[1] / "constraints" / "mnist"
sys.path.insert(0, str(EXAMPLES))

from generate_mnist_sum_tree import generate_mnist_sum_tree_dsl  # noqa: E402


def compile_with_fastpath(text: str, monkeypatch, enabled: bool):
    monkeypatch.setenv("SAT_FUNCTIONAL_FASTPATH", "1" if enabled else "0")
    return compile_constraints(parse_constraints(text))


def test_functional_tree_fastpath_is_artifact_identical(monkeypatch):
    text = generate_mnist_sum_tree_dsl(2)
    optimized = compile_with_fastpath(text, monkeypatch, True)
    general = compile_with_fastpath(text, monkeypatch, False)
    assert to_dsl(optimized) == to_dsl(general)


def test_conflicting_outputs_fall_back_and_keep_input_constraint(monkeypatch):
    # The same exact trigger a=0 requires y=0 and y=1.  This is not a
    # function; eliminating y must derive the real input restriction a=1.
    text = """
    var a 2
    var y 2
    ordering a y
    [1]:a or [0]:y
    [1]:a or [1]:y
    """
    optimized = compile_with_fastpath(text, monkeypatch, True)
    general = compile_with_fastpath(text, monkeypatch, False)
    assert to_dsl(optimized) == to_dsl(general)
    assert len(optimized.per_var_clauses[0]) == 1
    assert optimized.per_var_clauses[0][0].get(0) == frozenset({1})


def test_partial_deterministic_function_preserves_semantics(monkeypatch):
    # Only a=0 is constrained.  The fast path may prove zero resolvents, but
    # must retain the original bucket exactly so a=1 remains unconstrained.
    text = """
    var a 2
    var y 2
    ordering a y
    [1]:a or [0]:y
    """
    optimized = compile_with_fastpath(text, monkeypatch, True)
    general = compile_with_fastpath(text, monkeypatch, False)
    assert to_dsl(optimized) == to_dsl(general)
