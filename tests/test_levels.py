from pathlib import Path

import pytest
import torch

from constraints import Constraints, compute_levels


EXAMPLE1_PATH = Path(__file__).resolve().parents[1] / "examples" / "example1_shapes.txt"
SUDOKU_TXT = Path(__file__).resolve().parents[1] / "examples" / "sudoku_2x2.txt"


def test_compute_levels_no_edges_single_level():
    """Fully independent variables collapse into one level."""
    nbrs = {0: [], 1: [], 2: [], 3: []}
    assert compute_levels([0, 1, 2, 3], nbrs) == [[0, 1, 2, 3]]


def test_compute_levels_chain_is_sequential():
    """A -> B -> C dependency chain produces 3 levels of 1 each."""
    nbrs = {0: [], 1: [0], 2: [1]}
    assert compute_levels([0, 1, 2], nbrs) == [[0], [1], [2]]


def test_compute_levels_warcraft_3x3_anti_diagonals():
    """A 3x3 Warcraft-style grid (each cell depends on left + above) levelises
    into anti-diagonal sweeps."""
    N = 3
    cells = [(i, j) for i in range(N) for j in range(N)]
    flat = lambda c: c[0] * N + c[1]
    nbrs = {}
    for (i, j) in cells:
        deps = []
        if i > 0:
            deps.append(flat((i - 1, j)))
        if j > 0:
            deps.append(flat((i, j - 1)))
        nbrs[flat((i, j))] = deps

    levels = compute_levels([flat(c) for c in cells], nbrs)
    assert len(levels) == 5
    for k, level in enumerate(levels):
        for v in level:
            i, j = divmod(v, N)
            assert i + j == k, f"level {k} contains cell ({i},{j}) with i+j={i+j}"


def test_compute_levels_mnist_sum_two_levels():
    """N independent digits plus a sum that depends on all -> 2 levels."""
    N = 5
    nbrs = {i: [] for i in range(N)}
    nbrs[N] = list(range(N))
    levels = compute_levels(list(range(N + 1)), nbrs)
    assert levels == [list(range(N)), [N]]


def test_example1_is_fully_sequential():
    """Example 1 is a chain of implications: y_1 -> y_2 -> y_3."""
    c = Constraints.from_file(EXAMPLE1_PATH)
    assert c.ordering_levels == [[0], [1], [2]]


def test_sudoku_2x2_is_fully_sequential_after_compile():
    """Resolvents fan dependencies across every cell -> fully sequential."""
    c = Constraints.from_file(SUDOKU_TXT)
    assert c.ordering_levels == [[0], [1], [2], [3]]


def test_algebraic_mnist_sum_parallel_levels():
    """Constraints.from_function with explicit neighbours unlocks the
    digits-in-one-level + sum-in-the-next structure."""
    N = 8

    def sum_fn(step, prior, var_domain):
        if step < N:
            return frozenset(range(var_domain))
        return frozenset({sum(prior.values())})

    c = Constraints.from_function(
        sum_fn,
        var_domains=[10] * N + [9 * N + 1],
        neighbours={i: [] for i in range(N)} | {N: list(range(N))},
    )
    assert c.ordering_levels == [list(range(N)), [N]]


def test_algebraic_default_neighbours_is_fully_sequential():
    """Without explicit `neighbours`, every prior is assumed a dependency."""
    N = 3

    def trivial(step, prior, var_domain):
        return frozenset(range(var_domain))

    c = Constraints.from_function(trivial, var_domains=[3] * N)
    assert c.ordering_levels == [[0], [1], [2]]


def test_casper_layer_correct_on_two_level_mnist_sum():
    """End-to-end Theorem-4 check on a constraint with a multi-variable level:
    4 digits in level 0, sum in level 1."""
    from casper_layer import CasperLayer

    torch.manual_seed(0)
    N = 4

    def sum_fn(step, prior, var_domain):
        if step < N:
            return frozenset(range(var_domain))
        return frozenset({sum(prior.values())})

    var_ranges = [10] * N + [9 * N + 1]
    c = Constraints.from_function(
        sum_fn,
        var_domains=var_ranges,
        neighbours={i: [] for i in range(N)} | {N: list(range(N))},
    )
    layer = CasperLayer(c)

    assert layer.ordering_levels == [list(range(N)), [N]]

    B = 8
    parts = [torch.softmax(torch.randn(B, h), dim=1) for h in var_ranges]
    probs = torch.cat(parts, dim=1)
    adjusted, info = layer(probs)

    offsets = [0]
    for h in var_ranges[:-1]:
        offsets.append(offsets[-1] + h)
    for b in range(B):
        digits = [int(adjusted[b, offsets[i] : offsets[i] + 10].argmax().item())
                  for i in range(N)]
        s = int(adjusted[b, offsets[N] : offsets[N] + var_ranges[N]].argmax().item())
        assert s == sum(digits), (
            f"sample {b}: digits {digits} sum to {sum(digits)} but predicted s = {s}"
        )
