"""Unit tests for GroupKPassCasper (tasks/sudoku/local_casper.py).

Pure casper: needs only casper/ and tasks/sudoku/ on sys.path, the same entries
tests/conftest.py adds. Run from the repo root:

    python -m pytest -q tasks/sudoku/test_group_kpass.py

The parity test imports the reference implementation from llm-sudoku by file
path and is skipped when that checkout is absent.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for _sub in ("casper", "tasks/sudoku"):
    if str(ROOT / _sub) not in sys.path:
        sys.path.insert(0, str(ROOT / _sub))

from constraints import Constraints
from local_casper import GroupKPassCasper
from resolution import from_json, to_dsl

N = 9
GROUP_TXT = ROOT / "constraints" / "sudoku" / "sudoku_group9_compiled.txt"
GROUP_JSON = ROOT / "constraints" / "sudoku" / "sudoku_group9_compiled.json"

LLM_SUDOKU_SRC = Path("/home/la2324/llm-sudoku/sudoku/src")
LLM_SUDOKU_LAYER = LLM_SUDOKU_SRC / "casper_sudoku_layer.py"

LATIN = [(3 * (r % 3) + r // 3 + c) % 9 for r in range(N) for c in range(N)]


@pytest.fixture(scope="module")
def group_cs():
    return Constraints.from_compiled_dsl(GROUP_TXT)


def peaky(board, high=0.92):
    p = torch.full((1, 81, N), 0.01)
    p[0, torch.arange(81), torch.tensor(board)] = high
    return p / p.sum(dim=2, keepdim=True)


def one_duplicate():
    """Cell 1 (row 0) gets cell 0's digit, with lower confidence than its own."""
    p = peaky(LATIN)
    p[0, 1] = 0.01
    p[0, 1, LATIN[0]] = 0.80
    p[0, 1, LATIN[1]] = 0.15
    p[0, 1] = p[0, 1] / p[0, 1].sum()
    return p


def board_is_valid(grid):
    rows = [[grid[r * N + c] for c in range(N)] for r in range(N)]
    cols = [[grid[r * N + c] for r in range(N)] for c in range(N)]
    boxes = [
        [grid[(br + dr) * N + bc + dc] for dr in range(3) for dc in range(3)]
        for br in (0, 3, 6) for bc in (0, 3, 6)
    ]
    return all(sorted(g) == list(range(N)) for g in rows + cols + boxes)


def test_compiled_txt_matches_fresh_dsl():
    fresh = to_dsl(from_json(json.loads(GROUP_JSON.read_text())))
    assert GROUP_TXT.read_text() == fresh


def test_valid_board_zero_edits(group_cs):
    layer = GroupKPassCasper(group_cs, N, max_rounds=10)
    p = peaky(LATIN)
    out, info = layer(p.reshape(1, -1))
    assert info["edits_per_round"] == [[0]]
    assert info["certified_at"] == [1]
    assert info["rounds_used"] == 1
    assert torch.equal(out, p.reshape(1, -1))


def test_single_duplicate_repaired(group_cs):
    layer = GroupKPassCasper(group_cs, N, max_rounds=10)
    out, info = layer(one_duplicate().reshape(1, -1))
    grid = out.view(1, 81, N).argmax(dim=2)[0].tolist()
    assert board_is_valid(grid)
    assert grid == LATIN
    assert info["certified_at"][0] is not None
    assert info["final_assignments"][0] == LATIN


def test_gradient_flows(group_cs):
    layer = GroupKPassCasper(group_cs, N, max_rounds=10)
    p = one_duplicate().reshape(1, -1).clone().requires_grad_(True)
    out, info = layer(p)
    (-(out.clamp_min(1e-9)).log().mean()).backward()
    assert p.grad is not None and p.grad.abs().sum() > 0
    assert info["projection_loss"].requires_grad


def test_rejects_unknown_cell_order(group_cs):
    with pytest.raises(ValueError):
        GroupKPassCasper(group_cs, N, cell_order="random")


def _load_reference_class():
    """Import llm-sudoku's GroupKPassCasper by file path without leaving its
    sudoku/src modules (group_order_utils, ...) cached in sys.modules, and
    without writing bytecode into the reference checkout."""
    before = set(sys.modules)
    added = str(LLM_SUDOKU_SRC) not in sys.path
    if added:
        sys.path.insert(0, str(LLM_SUDOKU_SRC))
    dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(
            "_llm_sudoku_casper_sudoku_layer", LLM_SUDOKU_LAYER
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.GroupKPassCasper
    finally:
        sys.dont_write_bytecode = dont_write
        if added:
            sys.path.remove(str(LLM_SUDOKU_SRC))
        for name in set(sys.modules) - before:
            module_file = getattr(sys.modules[name], "__file__", None) or ""
            if module_file.startswith(str(LLM_SUDOKU_SRC)):
                del sys.modules[name]


@pytest.mark.skipif(not LLM_SUDOKU_LAYER.is_file(),
                    reason="llm-sudoku reference checkout not present")
@pytest.mark.parametrize("cell_order", ["confidence", "canonical"])
def test_bit_identical_to_llm_sudoku(group_cs, cell_order):
    Reference = _load_reference_class()
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(8, 81, N), dim=2).reshape(8, -1)

    ours = GroupKPassCasper(group_cs, N, max_rounds=10, cell_order=cell_order)
    ref = Reference(group_cs, N, max_rounds=10, cell_order=cell_order)
    out_a, info_a = ours(probs.clone())
    out_b, info_b = ref(probs.clone())

    assert torch.equal(out_a, out_b)
    assert torch.equal(info_a["projection_loss"], info_b["projection_loss"])
    for key in ("total_l2_distance", "rounds_used", "certified_at",
                "edits_per_round", "final_assignments"):
        assert info_a[key] == info_b[key], key
    # The batch must actually exercise the repair path.
    assert sum(sum(r) for r in info_a["edits_per_round"]) > 0


# ---- group_solver="search" -------------------------------------------------


def test_rejects_unknown_group_solver(group_cs):
    with pytest.raises(ValueError):
        GroupKPassCasper(group_cs, N, group_solver="hungarian")


def test_search_valid_board_zero_edits(group_cs):
    layer = GroupKPassCasper(group_cs, N, max_rounds=10, group_solver="search")
    p = peaky(LATIN)
    out, info = layer(p.reshape(1, -1))
    assert info["edits_per_round"] == [[0]]
    assert info["certified_at"] == [1]
    assert torch.equal(out, p.reshape(1, -1))


def test_search_single_duplicate_repaired(group_cs):
    layer = GroupKPassCasper(group_cs, N, max_rounds=10, group_solver="search")
    out, info = layer(one_duplicate().reshape(1, -1))
    grid = out.view(1, 81, N).argmax(dim=2)[0].tolist()
    assert grid == LATIN
    assert info["certified_at"][0] is not None
    assert info["final_assignments"][0] == LATIN


def test_search_gradient_flows(group_cs):
    layer = GroupKPassCasper(group_cs, N, max_rounds=10, group_solver="search")
    p = one_duplicate().reshape(1, -1).clone().requires_grad_(True)
    out, info = layer(p)
    (-(out.clamp_min(1e-9)).log().mean()).backward()
    assert p.grad is not None and p.grad.abs().sum() > 0
    assert info["projection_loss"].requires_grad


def test_search_moves_the_cell_with_the_plausible_alternative(group_cs):
    """Two cells hold digit 3 and digit 5 is free: the cell that finds 5 plausible moves.

    Greedy (least confident first) would move cell 1, the less confident
    holder of 3, onto a digit it gives 1%; the optimal assignment moves cell 0.
    """
    layer = GroupKPassCasper(group_cs, N, group_solver="search")
    probs = torch.full((N, N), 0.01)
    for k, d in enumerate([3, 3, 0, 1, 2, 4, 6, 7, 8]):
        probs[k, d] = 0.9
    probs[0] = 0.0125
    probs[0, 3], probs[0, 5] = 0.60, 0.35
    probs[1] = 0.0629
    probs[1, 3], probs[1, 5] = 0.55, 0.01
    cost = (-(probs / probs.sum(1, keepdim=True)).log()).tolist()
    assert layer._search_group(cost) == [5, 3, 0, 1, 2, 4, 6, 7, 8]


def test_search_matches_optimal_assignment(group_cs):
    """The compiled-walk search returns a minimum-cost permutation (checked with Hungarian)."""
    from scipy.optimize import linear_sum_assignment

    layer = GroupKPassCasper(group_cs, N, group_solver="search")
    gen = torch.Generator().manual_seed(0)
    for _ in range(20):
        probs = torch.softmax(3 * torch.randn(N, N, generator=gen), dim=1)
        cost = -probs.log()
        values = layer._search_group(cost.tolist())
        assert sorted(values) == list(range(N))
        rows, cols = linear_sum_assignment(cost.numpy())
        optimum = float(cost[rows, cols].sum())
        got = float(sum(cost[k, v] for k, v in enumerate(values)))
        assert abs(got - optimum) < 1e-5
