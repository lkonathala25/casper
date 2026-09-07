"""Compiled-constraint path for the MNIST benchmarks.

Covers the pipeline: generated DSL -> compile (or elimination-free
from_compiled_dsl load) -> CompiledConstraintLayer, and its bit-identical
equivalence to the hard-coded ConstraintLayer / CarryChainConstraintLayer.
"""

import sys
from pathlib import Path

import pytest
import torch

from signed_clauses import parse_constraints
from resolution import compile_constraints, enumerate_models, to_dsl
from compiled_layer import CompiledConstraintLayer
from mnist_sum_constraint_layer import ConstraintLayer
from carry_chain import compute_carry_chain
# Archived oracles: the hard-coded layers the compiled path is checked against.
from carry_chain_constraint_layer import CarryChainConstraintLayer

EXAMPLES = Path(__file__).resolve().parents[1] / "tasks" / "mnist"
CONSTRAINTS = Path(__file__).resolve().parents[1] / "constraints" / "mnist"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from generate_mnist_add import generate_mnist_add_dsl  # noqa: E402

import sys
from pathlib import Path as _P
_ARCHIVE = _P(__file__).resolve().parents[1] / "archive" / "mnist_closed_form"
if str(_ARCHIVE) not in sys.path:
    sys.path.append(str(_ARCHIVE))

SUM_N2 = CONSTRAINTS / "mnist_sum_n2.txt"
SUM_N2_COMPILED = CONSTRAINTS / "mnist_sum_n2_compiled.txt"
ADD_N1 = CONSTRAINTS / "mnist_add_n1.txt"
ADD_N2 = CONSTRAINTS / "mnist_add_n2.txt"
ADD_N2_COMPILED = CONSTRAINTS / "mnist_add_n2_compiled.txt"


def _random_probs(var_ranges: list[int], B: int) -> torch.Tensor:
    parts = [torch.softmax(torch.randn(B, h), dim=1) for h in var_ranges]
    return torch.cat(parts, dim=1)


def _flat_clauses(cc) -> list:
    return [c for pi_bar in cc.per_var_clauses for c in pi_bar]


# ---------------------------------------------------------------------------
# Compile artifacts
# ---------------------------------------------------------------------------

def test_sum_n2_artifact_matches_recompile():
    cs = parse_constraints(SUM_N2.read_text())
    cc = compile_constraints(cs)
    assert to_dsl(cc) == SUM_N2_COMPILED.read_text()


def test_add_n2_artifact_matches_recompile():
    cs = parse_constraints(ADD_N2.read_text())
    cc = compile_constraints(cs)
    assert to_dsl(cc) == ADD_N2_COMPILED.read_text()


def test_add_compile_is_resolvent_free():
    """The generated add DSL is its own compile fixpoint: elimination keeps
    zero resolvents, so the compiled clause set equals the input clause set.
    (This is what licenses loading big-N generated files directly via
    Constraints.from_compiled_dsl without running the compile.)"""
    cs = parse_constraints(ADD_N2.read_text())
    cc = compile_constraints(cs)
    assert set(_flat_clauses(cc)) == set(cs.clauses)


def test_sum_n2_lemma2_brute_force():
    """Models of the compiled set == models of the input set (1,900 assignments)."""
    cs = parse_constraints(SUM_N2.read_text())
    cc = compile_constraints(cs)
    models_in = enumerate_models(cs.clauses, cs.var_domains)
    models_out = enumerate_models(_flat_clauses(cc), cc.var_domains)
    assert models_in == models_out
    # Exactly one model per digit pair: S forced to the sum.
    assert len(models_in) == 100
    assert all(y1 + y2 == s for y1, y2, s in models_in)


def test_add_n1_lemma2_brute_force():
    """Models of the compiled add N=1 set (10,000 assignments over
    a_1, b_1, s_0, s_1)."""
    cs = parse_constraints(ADD_N1.read_text())
    cc = compile_constraints(cs)
    models_in = enumerate_models(cs.clauses, cs.var_domains)
    models_out = enumerate_models(_flat_clauses(cc), cc.var_domains)
    assert models_in == models_out
    assert len(models_in) == 100
    assert all(
        s0 * 10 + s1 == a + b for a, b, s0, s1 in models_in
    )


# ---------------------------------------------------------------------------
# Layer equivalence with the hard-coded layers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sum_n2_equals_hardcoded_layer(seed: int):
    torch.manual_seed(seed)
    var_ranges = [10, 10, 19]
    legacy = ConstraintLayer(variable_ranges=var_ranges)
    compiled = CompiledConstraintLayer(SUM_N2_COMPILED, var_ranges)

    probs = _random_probs(var_ranges, B=16)
    legacy_out, legacy_info = legacy(probs.clone())
    compiled_out, compiled_info = compiled(probs.clone())

    assert torch.allclose(legacy_out, compiled_out, atol=0.0, rtol=0.0)
    assert legacy_info["total_constraints_applied"] == \
        compiled_info["total_constraints_applied"]
    assert legacy_info["total_l2_distance"] == \
        pytest.approx(compiled_info["total_l2_distance"])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_add_n2_equals_hardcoded_layer(seed: int):
    torch.manual_seed(seed)
    n = 2
    var_ranges = [10] * (2 * n) + [10] * (n + 1)
    legacy = CarryChainConstraintLayer(n)
    compiled = CompiledConstraintLayer(ADD_N2_COMPILED, var_ranges)

    probs = _random_probs(var_ranges, B=16)
    legacy_out, legacy_info = legacy(probs.clone())
    compiled_out, compiled_info = compiled(probs.clone())

    assert torch.allclose(legacy_out, compiled_out, atol=0.0, rtol=0.0)
    assert torch.allclose(
        legacy_info["projection_loss"], compiled_info["projection_loss"],
        atol=0.0, rtol=0.0,
    )
    assert legacy_info["total_constraints_applied"] == \
        compiled_info["total_constraints_applied"]


@pytest.mark.parametrize("seed", [0, 1])
def test_add_n3_generated_dsl_direct_load_equals_hardcoded(seed: int, tmp_path):
    """The big-N path: load the *generated* DSL via from_compiled_dsl (no
    compile — it is its own fixpoint) and match the hard-coded layer."""
    torch.manual_seed(seed)
    n = 3
    dsl_path = tmp_path / "mnist_add_n3.txt"
    dsl_path.write_text(generate_mnist_add_dsl(n))

    var_ranges = [10] * (2 * n) + [10] * (n + 1)
    legacy = CarryChainConstraintLayer(n)
    compiled = CompiledConstraintLayer(dsl_path, var_ranges)

    probs = _random_probs(var_ranges, B=16)
    legacy_out, _ = legacy(probs.clone())
    compiled_out, _ = compiled(probs.clone())

    assert torch.allclose(legacy_out, compiled_out, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("n", [2, 3])
def test_add_infer_matches_compute_carry_chain(n: int, tmp_path):
    """Output-digit argmaxes after projection == the carry chain of the input
    argmaxes; input slices pass through untouched."""
    torch.manual_seed(7)
    dsl_path = tmp_path / f"mnist_add_n{n}.txt"
    dsl_path.write_text(generate_mnist_add_dsl(n))

    var_ranges = [10] * (2 * n) + [10] * (n + 1)
    compiled = CompiledConstraintLayer(dsl_path, var_ranges)

    B = 12
    probs = _random_probs(var_ranges, B)
    adjusted, _ = compiled(probs.clone())

    input_width = 2 * n * 10
    assert torch.equal(adjusted[:, :input_width], probs[:, :input_width])

    input_probs = probs[:, :input_width].view(B, 2 * n, 10)
    a_preds = input_probs[:, :n].argmax(dim=2)
    b_preds = input_probs[:, n:].argmax(dim=2)
    s_targets = compute_carry_chain(a_preds, b_preds)

    output_probs = adjusted[:, input_width:].view(B, n + 1, 10)
    assert torch.equal(output_probs.argmax(dim=2), s_targets)


# ---------------------------------------------------------------------------
# Adapter mechanics
# ---------------------------------------------------------------------------

def test_autograd_backward_through_compiled_layer():
    torch.manual_seed(0)
    n = 2
    var_ranges = [10] * (2 * n) + [10] * (n + 1)
    compiled = CompiledConstraintLayer(ADD_N2_COMPILED, var_ranges)

    probs = _random_probs(var_ranges, B=4).clone().requires_grad_(True)
    adjusted, info = compiled(probs)
    loss = adjusted.sum() + info["projection_loss"]
    loss.backward()

    assert probs.grad is not None
    assert not torch.isnan(probs.grad).any()
    assert probs.grad.abs().sum().item() > 0


def test_l2_per_var_consistent_with_totals():
    """CasperLayer's new per-variable breakdown sums to the existing totals."""
    torch.manual_seed(3)
    n = 2
    var_ranges = [10] * (2 * n) + [10] * (n + 1)
    compiled = CompiledConstraintLayer(ADD_N2_COMPILED, var_ranges)

    probs = _random_probs(var_ranges, B=16)
    _, casper_info = compiled.casper(
        torch.cat(
            [probs] + [
                torch.full((16, h), 1.0 / h) for h in compiled.aux_domains
            ],
            dim=1,
        )
    )
    l2_per_var = casper_info["l2_per_var"]
    assert l2_per_var.shape == (16, len(compiled.constraints.var_domains))
    assert l2_per_var.sum().item() == \
        pytest.approx(casper_info["total_l2_distance"])
    assert sum(casper_info["n_projected_per_var"]) == \
        casper_info["total_constraints_applied"]


def test_domain_mismatch_rejected():
    with pytest.raises(ValueError, match="leading domains"):
        CompiledConstraintLayer(ADD_N2_COMPILED, [10, 10, 19])
