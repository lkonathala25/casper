from pathlib import Path

import pytest
import torch

from compiled_layer import CompiledConstraintLayer
from functional_compiled_layer import FunctionalCompiledConstraintLayer


CONSTRAINTS = Path(__file__).resolve().parents[1] / "constraints" / "mnist"


def _probs(batch, n_vars, scale, generator):
    """Per-variable softmax over 10 categories; scale 0 gives exact ties."""
    logits = torch.randn(batch, n_vars, 10, generator=generator) * scale
    return torch.softmax(logits, dim=2).reshape(batch, n_vars * 10)


@pytest.mark.parametrize("file_name, n_vars", [
    ("mnist_add_n2_compiled.txt", 7),
    ("mnist_add_n4.txt", 13),
    ("mnist_sum_digit_tree_n4.txt", 6),
    ("mnist_sum_digit_tree_n16.txt", 18),
])
@pytest.mark.parametrize("scale", [0.0, 1e-3, 0.5, 6.0])
def test_batched_layer_matches_generic_walk(file_name, n_vars, scale):
    """Outputs, projection loss, counts and input gradients are bit-identical."""
    generator = torch.Generator().manual_seed(7)
    variable_ranges = [10] * n_vars
    generic = CompiledConstraintLayer(CONSTRAINTS / file_name, variable_ranges)
    batched = FunctionalCompiledConstraintLayer(CONSTRAINTS / file_name, variable_ranges)
    for batch in (1, 5, 32):
        probs = _probs(batch, n_vars, scale, generator)
        weights = torch.randn(batch, n_vars * 10, generator=generator)
        results = []
        for layer in (generic, batched):
            x = probs.clone().requires_grad_()
            out, info = layer(x)
            ((out * weights).sum() + info["projection_loss"]).backward()
            results.append((out.detach(), info, x.grad))
        (out_g, info_g, grad_g), (out_b, info_b, grad_b) = results
        assert torch.equal(out_b, out_g)
        assert torch.equal(grad_b, grad_g)
        assert torch.equal(info_b["projection_loss"].detach(), info_g["projection_loss"].detach())
        assert info_b["total_constraints_applied"] == info_g["total_constraints_applied"]
        assert info_b["avg_l2_per_sample"] == info_g["avg_l2_per_sample"]


@pytest.mark.parametrize("file_name, n_vars", [
    ("mnist_add_n2_compiled.txt", 7),
    ("mnist_sum_digit_tree_n4.txt", 6),
])
@pytest.mark.parametrize("scale", [0.5, 6.0])
def test_batched_cvxpy_matches_generic_cvxpy(file_name, n_vars, scale):
    """The batched layer with cvxpy projects as the per-sample cvxpy walk does, to solver tolerance."""
    pytest.importorskip("cvxpylayers")
    generator = torch.Generator().manual_seed(11)
    variable_ranges = [10] * n_vars
    generic = CompiledConstraintLayer(CONSTRAINTS / file_name, variable_ranges,
                                      projection_backend="cvxpy")
    batched = FunctionalCompiledConstraintLayer(CONSTRAINTS / file_name, variable_ranges,
                                                projection_backend="cvxpy")
    for batch in (1, 16):
        probs = _probs(batch, n_vars, scale, generator)
        out_g, info_g = generic(probs.clone())
        out_b, info_b = batched(probs.clone())
        assert torch.allclose(out_b, out_g, atol=1e-4)
        assert info_b["total_constraints_applied"] == info_g["total_constraints_applied"]
        assert abs(info_b["avg_l2_per_sample"] - info_g["avg_l2_per_sample"]) < 1e-4


def test_cvxpy_project_rows_solves_each_row_exactly():
    """Each row meets its own target's margin to SCS tolerance, never above the closed form's L2."""
    pytest.importorskip("cvxpylayers")
    from cvxpy_minedit import CVXPYMinEditL2
    from minedit import MinEditL2
    generator = torch.Generator().manual_seed(3)
    p = torch.softmax(torch.randn(64, 10, generator=generator) * 3, dim=1)
    target = torch.randint(0, 10, (64,), generator=generator)
    exact, l2_exact = CVXPYMinEditL2(10).project_rows(p, target)
    closed, l2_closed = MinEditL2(10).project_rows(p, target)
    others = torch.ones_like(p, dtype=torch.bool).scatter_(1, target.unsqueeze(1), False)
    margin = exact.gather(1, target.unsqueeze(1)) - exact
    assert bool((margin[others] >= 1e-3 - 2.5e-4).all())
    assert bool((l2_exact <= l2_closed + 1e-4).all())
