from pathlib import Path

import pytest
import torch

from signed_clauses import parse_constraints, constraints_satisfied
from constraints import Constraints
from casper_layer import CasperLayer


EXAMPLE1_PATH = Path(__file__).resolve().parents[1] / "examples" / "example1_shapes.txt"


def _example1_constraints() -> Constraints:
    return Constraints.from_file(EXAMPLE1_PATH)


def test_shape_smoke():
    """Forward runs, output shape matches input, each per-variable slice
    still sums to 1, projection_loss is a tensor."""
    c = _example1_constraints()
    layer = CasperLayer(c)
    B = 4
    total = sum(c.var_domains)
    logits = torch.randn(B, total)
    offset = 0
    parts = []
    for h in c.var_domains:
        parts.append(torch.softmax(logits[:, offset : offset + h], dim=1))
        offset += h
    probs = torch.cat(parts, dim=1)

    adjusted, info = layer(probs)

    assert adjusted.shape == probs.shape
    offset = 0
    for h in c.var_domains:
        sums = adjusted[:, offset : offset + h].sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        offset += h

    assert "projection_loss" in info
    assert isinstance(info["projection_loss"], torch.Tensor)


def test_identity_no_constraint():
    """Unrestricted feasibility -> layer is a no-op."""
    def trivial(step, prior, var_domain):
        return frozenset(range(var_domain))

    c = Constraints.from_function(trivial, var_domains=[3, 4, 2])
    layer = CasperLayer(c)

    probs = torch.softmax(torch.randn(5, 3 + 4 + 2), dim=1)
    parts = []
    for slc, h in [(slice(0, 3), 3), (slice(3, 7), 4), (slice(7, 9), 2)]:
        parts.append(torch.softmax(probs[:, slc], dim=1))
    probs = torch.cat(parts, dim=1)

    adjusted, info = layer(probs)
    assert torch.equal(adjusted, probs)
    assert info["total_constraints_applied"] == 0


def test_argmax_already_feasible_no_projection():
    """Probs whose argmaxes already land in the feasible set pass through
    unchanged."""

    def singleton(step, prior, var_domain):
        if step == 0:
            return frozenset(range(var_domain))
        return frozenset({1})

    c = Constraints.from_function(singleton, var_domains=[3, 3])
    layer = CasperLayer(c)

    p0 = torch.tensor([[0.1, 0.2, 0.7]])
    p1 = torch.tensor([[0.2, 0.7, 0.1]])
    probs = torch.cat([p0, p1], dim=1)

    adjusted, info = layer(probs)
    assert torch.equal(adjusted, probs)
    assert info["total_constraints_applied"] == 0


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_theorem4_example1_correctness(seed: int):
    """For random inputs, CasperLayer's output argmaxes satisfy the original Π."""
    torch.manual_seed(seed)

    text = EXAMPLE1_PATH.read_text()
    cs = parse_constraints(text)
    layer = CasperLayer(Constraints.from_text(text))

    B = 16
    parts = []
    for h in layer.var_domains:
        parts.append(torch.softmax(torch.randn(B, h), dim=1))
    probs = torch.cat(parts, dim=1)

    adjusted, info = layer(probs)

    offsets = [0]
    for h in layer.var_domains[:-1]:
        offsets.append(offsets[-1] + h)
    for b in range(B):
        assignment = {}
        for v, (off, h) in enumerate(zip(offsets, layer.var_domains)):
            assignment[v] = int(adjusted[b, off : off + h].argmax().item())
        assert constraints_satisfied(cs.clauses, assignment), (
            f"seed={seed} sample={b}: assignment {assignment} violates Π"
        )


def test_candidate_enumeration_picks_higher_prob_feasible():
    """Non-singleton feasibility: argmin_t over t ∈ S_i picks the closer
    feasible candidate."""
    text = """
    var y_1 4
    var y_2 4
    [0,1]:y_1 or [0,1]:y_2
    """
    layer = CasperLayer(Constraints.from_text(text))

    p1 = torch.tensor([[0.1, 0.2, 0.4, 0.3]])
    p2 = torch.tensor([[0.1, 0.3, 0.2, 0.4]])
    probs = torch.cat([p1, p2], dim=1)

    adjusted, info = layer(probs)
    assert torch.equal(adjusted[:, :4], p1)
    assert int(adjusted[0, 4:].argmax().item()) == 1
    assert info["total_constraints_applied"] == 1


def _make_sum_feasibility_fn(n_digits: int):
    def fn(step, prior, var_domain):
        if step < n_digits:
            return frozenset(range(var_domain))
        return frozenset({sum(prior.values())})

    return fn


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_algebraic_sum_equals_existing_constraint_layer(seed: int):
    """CasperLayer with Constraints.from_function wrapping sum produces
    bit-identical projections to the legacy ConstraintLayer."""
    from constraint_layer import ConstraintLayer

    torch.manual_seed(seed)

    n_digits = 2
    var_ranges = [10] * n_digits + [9 * n_digits + 1]

    legacy = ConstraintLayer(variable_ranges=var_ranges)

    c = Constraints.from_function(
        _make_sum_feasibility_fn(n_digits), var_domains=var_ranges
    )
    casper = CasperLayer(c)

    B = 8
    parts = []
    for h in var_ranges:
        parts.append(torch.softmax(torch.randn(B, h), dim=1))
    probs = torch.cat(parts, dim=1)

    legacy_out, _ = legacy(probs.clone())
    casper_out, _ = casper(probs.clone())

    assert torch.allclose(legacy_out, casper_out, atol=0.0, rtol=0.0), (
        f"seed={seed}: divergence between legacy ConstraintLayer and "
        f"CasperLayer(Constraints.from_function(sum))"
    )


def test_autograd_backward():
    """projection_loss is differentiable; backward populates input.grad."""
    cf = _example1_constraints()
    layer = CasperLayer(cf)

    B = 4
    parts = []
    for h in cf.var_domains:
        parts.append(torch.softmax(torch.randn(B, h), dim=1))
    probs = torch.cat(parts, dim=1).clone().requires_grad_(True)

    adjusted, info = layer(probs)

    loss = adjusted.sum() + info["projection_loss"]
    loss.backward()

    assert probs.grad is not None
    assert not torch.isnan(probs.grad).any()
    assert probs.grad.abs().sum().item() > 0
