from itertools import product
from pathlib import Path

import pytest
import torch

from signed_clauses import constraints_satisfied, parse_constraints
from resolution import compile_constraints, enumerate_models
from constraints import Constraints
from casper_layer import CasperLayer


SUDOKU_TXT = Path(__file__).resolve().parents[1] / "constraints" / "sudoku" / "sudoku_2x2.txt"
SUDOKU_COMPILED = Path(__file__).resolve().parents[1] / "constraints" / "sudoku" / "sudoku_2x2_compiled.txt"


SOLUTIONS = {(0, 1, 1, 0), (1, 0, 0, 1)}


def test_sudoku_compiled_txt_matches_recompile():
    """Recompiling the committed compiled .txt yields the same result as
    compiling the original input (COMPILE idempotency)."""
    cs_input = parse_constraints(SUDOKU_TXT.read_text())
    fresh = compile_constraints(cs_input)

    cs_artifact = parse_constraints(SUDOKU_COMPILED.read_text())
    on_disk = compile_constraints(cs_artifact)

    assert on_disk.var_names == fresh.var_names
    assert on_disk.var_domains == fresh.var_domains
    assert on_disk.ordering == fresh.ordering
    assert on_disk.per_var_clauses == fresh.per_var_clauses


def test_sudoku_compile_produces_expected_clause_counts():
    """8 input clauses become 10 compiled clauses (8 originals + 2 resolvents)."""
    cs = parse_constraints(SUDOKU_TXT.read_text())
    cc = compile_constraints(cs)

    assert len(cs.clauses) == 8
    total = sum(len(pi_bar) for pi_bar in cc.per_var_clauses)
    assert total == 10

    counts = [len(pi_bar) for pi_bar in cc.per_var_clauses]
    assert counts == [0, 2, 4, 4]


def test_sudoku_lemma2_brute_force():
    """Brute-force confirms both the input and compiled clause sets admit
    exactly the 2 Latin-square models."""
    cs = parse_constraints(SUDOKU_TXT.read_text())
    cc = compile_constraints(cs)

    input_models = set(enumerate_models(cs.clauses, cs.var_domains))
    all_compiled = [c for pi_bar in cc.per_var_clauses for c in pi_bar]
    compiled_models = set(enumerate_models(all_compiled, cc.var_domains))

    assert input_models == SOLUTIONS
    assert compiled_models == SOLUTIONS


def test_sudoku_infer_walks_the_solution():
    """At each INFER step the feasible set under the chain of prior argmaxes
    converges to the unique Latin-square completion."""
    c = Constraints.from_file(SUDOKU_COMPILED)

    assert c.feasible_set(0, {}) == frozenset({0, 1})
    assert c.feasible_set(1, {0: 0}) == frozenset({1})
    assert c.feasible_set(2, {0: 0, 1: 1}) == frozenset({1})
    assert c.feasible_set(3, {0: 0, 1: 1, 2: 1}) == frozenset({0})

    assert c.feasible_set(1, {0: 1}) == frozenset({0})
    assert c.feasible_set(2, {0: 1, 1: 0}) == frozenset({0})
    assert c.feasible_set(3, {0: 1, 1: 0, 2: 0}) == frozenset({1})


def _argmax_assignment(adjusted: torch.Tensor, sample: int) -> dict[int, int]:
    return {
        v: int(adjusted[sample, v * 2 : v * 2 + 2].argmax().item())
        for v in range(4)
    }


def test_sudoku_layer_projects_to_a_valid_latin_square():
    """A constraint-violating input is projected into one of the 2 Latin squares."""
    cs = parse_constraints(SUDOKU_TXT.read_text())
    layer = CasperLayer(Constraints.from_file(SUDOKU_COMPILED))

    p = torch.tensor([[0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3]])
    pre = _argmax_assignment(p, 0)
    assert pre == {0: 0, 1: 0, 2: 0, 3: 0}
    assert not constraints_satisfied(cs.clauses, pre)

    adjusted, info = layer(p)
    post = _argmax_assignment(adjusted, 0)

    assert constraints_satisfied(cs.clauses, post), (
        f"projected assignment {post} still violates Pi"
    )
    flat = tuple(post[v] for v in range(4))
    assert flat in SOLUTIONS, f"projected assignment {flat} is not a Latin square"


@pytest.mark.parametrize("seed", range(8))
def test_sudoku_layer_theorem4_random(seed: int):
    """For random softmax inputs, the projected assignment is always a Latin square."""
    torch.manual_seed(seed)
    cs = parse_constraints(SUDOKU_TXT.read_text())
    layer = CasperLayer(Constraints.from_file(SUDOKU_COMPILED))

    B = 32
    parts = [torch.softmax(torch.randn(B, 2), dim=1) for _ in range(4)]
    probs = torch.cat(parts, dim=1)

    adjusted, _ = layer(probs)
    for b in range(B):
        post = _argmax_assignment(adjusted, b)
        flat = tuple(post[v] for v in range(4))
        assert constraints_satisfied(cs.clauses, post), (
            f"seed={seed} sample={b}: {post} violates Pi"
        )
        assert flat in SOLUTIONS, (
            f"seed={seed} sample={b}: {flat} not a valid Latin square"
        )


def test_sudoku_infer_consistency_with_brute_force():
    """For every reachable prior, the feasible set equals the brute-force
    set of values that extend the prior to a Latin square."""
    c = Constraints.from_file(SUDOKU_COMPILED)
    cs = parse_constraints(SUDOKU_TXT.read_text())

    input_models = set(enumerate_models(cs.clauses, cs.var_domains))

    for step in range(4):
        var = c.ordering[step]
        prior_vars = c.ordering[:step]
        for prior_values in product(*(range(c.var_domains[v]) for v in prior_vars)):
            prior = dict(zip(prior_vars, prior_values))
            feasible = c.feasible_set(step, prior)

            extending = set()
            for model in input_models:
                if all(model[v] == prior[v] for v in prior):
                    extending.add(model[var])

            if not extending:
                continue

            assert feasible == extending, (
                f"step={step} var={var} prior={prior}:\n"
                f"  INFER:       {sorted(feasible)}\n"
                f"  brute-force: {sorted(extending)}"
            )
