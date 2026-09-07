from pathlib import Path

import pytest
import torch

from signed_clauses import (
    SignedClause,
    SignedLiteral,
    constraints_satisfied,
    parse_constraints,
)
from resolution import compile_constraints
from constraints import Constraints
from casper_layer import CasperLayer


DEMO_TXT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "resolvent_demo.txt"
DEMO_COMPILED = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "resolvent_demo_compiled.txt"


def _lit(var: int, *values: int) -> SignedLiteral:
    return SignedLiteral(var, frozenset(values))


def _clause(*lits: SignedLiteral) -> SignedClause:
    return SignedClause.from_literals(lits)


def test_compile_derives_expected_resolvent():
    """COMPILE on the demo file produces Pi_bar_2 = [resolvent] not in the input."""
    cs = parse_constraints(DEMO_TXT.read_text())
    cc = compile_constraints(cs)

    c1_original = _clause(_lit(0, 0), _lit(2, 0))
    c2_original = _clause(_lit(1, 0), _lit(2, 1))
    expected_resolvent = _clause(_lit(0, 0), _lit(1, 0))

    assert cc.per_var_clauses[0] == []
    assert cc.per_var_clauses[1] == [expected_resolvent]
    assert cc.per_var_clauses[2] == [c1_original, c2_original]

    assert expected_resolvent not in cs.clauses


def test_compiled_txt_artifact_matches_recompile():
    """The committed compiled .txt re-compiles to the same CompiledConstraints
    as the original input (COMPILE idempotency)."""
    cs_input = parse_constraints(DEMO_TXT.read_text())
    fresh = compile_constraints(cs_input)

    cs_artifact = parse_constraints(DEMO_COMPILED.read_text())
    on_disk = compile_constraints(cs_artifact)

    assert on_disk.var_names == fresh.var_names
    assert on_disk.var_domains == fresh.var_domains
    assert on_disk.ordering == fresh.ordering
    assert on_disk.per_var_clauses == fresh.per_var_clauses


def test_constraints_from_json_round_trip():
    c = Constraints.from_file(DEMO_COMPILED)
    assert c.var_domains == [2, 2, 2]
    assert c.ordering == [0, 1, 2]

    assert c.feasible_set(0, {}) == frozenset({0, 1})
    assert c.feasible_set(1, {0: 1}) == frozenset({0})
    assert c.feasible_set(1, {0: 0}) == frozenset({0, 1})
    assert c.feasible_set(2, {0: 1, 1: 0}) == frozenset({0})
    assert c.feasible_set(2, {0: 0, 1: 1}) == frozenset({1})


def test_casper_layer_from_compiled_json_enforces_pi():
    """Load the compiled file, wrap in CasperLayer, feed a Π-violating input,
    and assert the projected output's argmax satisfies every original clause."""
    cs = parse_constraints(DEMO_TXT.read_text())
    layer = CasperLayer(Constraints.from_file(DEMO_COMPILED))

    p_y1 = torch.tensor([[0.3, 0.7]])
    p_y2 = torch.tensor([[0.4, 0.6]])
    p_y3 = torch.tensor([[0.55, 0.45]])
    probs = torch.cat([p_y1, p_y2, p_y3], dim=1)

    pre_assignment = {
        0: int(p_y1.argmax(dim=1).item()),
        1: int(p_y2.argmax(dim=1).item()),
        2: int(p_y3.argmax(dim=1).item()),
    }
    assert pre_assignment == {0: 1, 1: 1, 2: 0}
    assert not constraints_satisfied(cs.clauses, pre_assignment)

    adjusted, info = layer(probs)

    post_assignment = {
        0: int(adjusted[0, :2].argmax().item()),
        1: int(adjusted[0, 2:4].argmax().item()),
        2: int(adjusted[0, 4:6].argmax().item()),
    }
    assert constraints_satisfied(cs.clauses, post_assignment), (
        f"projected assignment {post_assignment} still violates Pi"
    )
    assert info["total_constraints_applied"] >= 1


@pytest.mark.parametrize("seed", range(8))
def test_pipeline_theorem4_random(seed: int):
    """For random softmax inputs, projected outputs always satisfy Π."""
    torch.manual_seed(seed)
    cs = parse_constraints(DEMO_TXT.read_text())
    layer = CasperLayer(Constraints.from_file(DEMO_COMPILED))

    B = 32
    parts = [torch.softmax(torch.randn(B, 2), dim=1) for _ in range(3)]
    probs = torch.cat(parts, dim=1)

    adjusted, _ = layer(probs)
    for b in range(B):
        assignment = {
            0: int(adjusted[b, 0:2].argmax().item()),
            1: int(adjusted[b, 2:4].argmax().item()),
            2: int(adjusted[b, 4:6].argmax().item()),
        }
        assert constraints_satisfied(cs.clauses, assignment), (
            f"seed={seed} sample={b}: assignment {assignment} violates Pi"
        )
