from pathlib import Path

import torch

from compiled_layer import CompiledConstraintLayer
from functional_compiled_layer import FunctionalCompiledConstraintLayer


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
CONSTRAINTS = Path(__file__).resolve().parents[1] / "constraints" / "mnist"


def test_functional_compiled_matches_generic_add_n2():
    torch.manual_seed(7)
    variable_ranges = [10] * 7
    generic = CompiledConstraintLayer(
        CONSTRAINTS / "mnist_add_n2_compiled.txt", variable_ranges
    )
    functional = FunctionalCompiledConstraintLayer(
        CONSTRAINTS / "mnist_add_n2_compiled.txt", variable_ranges
    )
    probs = torch.cat([
        torch.softmax(torch.randn(16, domain), dim=1)
        for domain in variable_ranges
    ], dim=1)

    generic_out, generic_info = generic(probs)
    functional_out, functional_info = functional(probs)

    assert torch.equal(functional_out, generic_out)
    assert functional_info["total_constraints_applied"] == (
        generic_info["total_constraints_applied"]
    )
    assert torch.equal(
        functional_info["projection_loss"],
        generic_info["projection_loss"],
    )
