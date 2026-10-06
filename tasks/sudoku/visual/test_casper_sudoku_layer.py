"""Tests for the CASPER edit layer (casper_sudoku_layer.py).

Run from the repo root:  python -m pytest -q tasks/sudoku/visual/test_casper_sudoku_layer.py

visual/ is not on tests/conftest.py's path: its module names (datasets, models,
utils, ...) would shadow other tasks. Following tests/_warcraft_import.py, it is
added only for the duration of the import below, and the visual/ modules that
import pulled in are evicted from sys.modules afterwards.
"""
import importlib
import sys
from pathlib import Path

import pytest
import torch

VISUAL_DIR = Path(__file__).resolve().parent


def _load_visual(name):
    path = str(VISUAL_DIR)
    added = path not in sys.path
    if added:
        sys.path.insert(0, path)
    before = set(sys.modules)
    try:
        return importlib.import_module(name)
    finally:
        if added:
            sys.path.remove(path)
        for mod in set(sys.modules) - before:
            module_file = getattr(sys.modules[mod], '__file__', None) or ''
            if module_file.startswith(path):
                del sys.modules[mod]


_layer = _load_visual('casper_sudoku_layer')
CASPER_ROOT = _layer.CASPER_ROOT
DEFAULT_CONSTRAINTS_FILE = _layer.DEFAULT_CONSTRAINTS_FILE
CasperSudokuLayer = _layer.CasperSudokuLayer
load_group_constraints = _layer.load_group_constraints
validate_group_constraints = _layer.validate_group_constraints
from constraints import Constraints  # noqa: E402  (casper/, appended by casper_sudoku_layer)

GROUP_JSON = CASPER_ROOT / 'constraints' / 'sudoku' / 'sudoku_group9_compiled.json'
DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA not available'))]


def latin_square():
    return torch.tensor(
        [(3 * (r % 3) + r // 3 + c) % 9 for r in range(9) for c in range(9)]
    )


def peaky(digits, high=0.92):
    p = torch.full((1, 81, 9), 0.01)
    p[0, torch.arange(81), digits] = high
    return p / p.sum(dim=2, keepdim=True)


def test_txt_matches_json():
    a = load_group_constraints(DEFAULT_CONSTRAINTS_FILE)
    b = load_group_constraints(GROUP_JSON)
    for v in range(9):
        assert a.feasible_set(1, {0: v}) == b.feasible_set(1, {0: v})
    for step in range(9):
        prefix = {i: i for i in range(step)}
        assert a.feasible_set(step, prefix) == b.feasible_set(step, prefix)


def test_auto_rejects_raw_dsl_txt(tmp_path):
    raw = tmp_path / 'group.txt'
    raw.write_text(DEFAULT_CONSTRAINTS_FILE.read_text())
    with pytest.raises(ValueError, match='constraints-format'):
        load_group_constraints(raw)


def test_validation_rejects_non_alldifferent():
    # 9 vars of domain 9 with no constraints: every digit allowed next to v.
    free = Constraints.from_function(lambda step, prior, h: range(h), [9] * 9)
    with pytest.raises(ValueError, match='ALLDIFFERENT'):
        validate_group_constraints(free)
    small = Constraints.from_function(lambda step, prior, h: range(h), [4] * 4)
    with pytest.raises(ValueError, match='domains'):
        validate_group_constraints(small)


@pytest.mark.parametrize('device', DEVICES)
def test_shape_and_distribution(device):
    layer = CasperSudokuLayer().to(device)
    probs = torch.softmax(torch.randn(3, 81, 9), dim=2).to(device)
    out, info = layer(probs, max_iterations=3)
    assert out.shape == (3, 81, 9) and out.device == probs.device
    assert torch.allclose(out.sum(dim=2), torch.ones(3, 81, device=device), atol=1e-5)
    assert (out >= -1e-6).all()
    for key in ('n_iterations', 'converged', 'n_certified', 'total_constraints_applied',
                'avg_l2_per_sample', 'projection_loss'):
        assert key in info


def test_identity_on_valid_board():
    layer = CasperSudokuLayer()
    p = peaky(latin_square())
    out, info = layer(p)
    assert info['total_constraints_applied'] == 0
    assert info['converged']
    assert torch.equal(out.argmax(dim=2), p.argmax(dim=2))
    assert layer.constraint_satisfaction_rate(out) == 1.0


def _one_conflict():
    ls = latin_square()
    p = peaky(ls)
    # Cell 1 (row 0) gets cell 0's digit with lower confidence.
    p[0, 1] = 0.01
    p[0, 1, ls[0]] = 0.80
    p[0, 1, ls[1]] = 0.15
    p[0, 1] = p[0, 1] / p[0, 1].sum()
    return ls, p


def test_repairs_single_conflict():
    ls, p = _one_conflict()
    layer = CasperSudokuLayer()
    out, info = layer(p)
    assert layer.constraint_satisfaction_rate(out) == 1.0
    assert info['converged']
    assert torch.equal(out.argmax(dim=2)[0], ls)


def test_gradient_flows():
    _, p = _one_conflict()
    p = p.clone().requires_grad_(True)
    out, info = CasperSudokuLayer()(p)
    (-(out.clamp_min(1e-9)).log().mean()).backward()
    assert p.grad is not None and p.grad.abs().sum() > 0
    assert info['projection_loss'].requires_grad


def test_group_order():
    ls, p = _one_conflict()
    layer = CasperSudokuLayer(group_order=list(range(26, -1, -1)))
    out, info = layer(p)
    assert info['converged'] and layer.constraint_satisfaction_rate(out) == 1.0
    with pytest.raises(ValueError):
        CasperSudokuLayer(group_order=[0] * 27)


def test_no_state_added_to_checkpoints():
    assert len(CasperSudokuLayer().state_dict()) == 0
