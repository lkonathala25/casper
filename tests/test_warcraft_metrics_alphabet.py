import contextlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

# src/ and src_warcraft/ share module names (data, model, minedit, ...). Another
# test module may already have imported the src/ versions, so bind the warcraft
# ones under an isolated sys.path/sys.modules and restore afterwards.
_COLLIDING = ("data", "model", "minedit", "constraint_layer", "node_categories",
              "masked_loss", "edge_bridge", "lazy_exact_layer", "compute_metrics",
              "constraints", "signed_clauses", "resolution")


@contextlib.contextmanager
def _warcraft_first():
    saved_mods = {n: sys.modules.pop(n) for n in _COLLIDING if n in sys.modules}
    saved_path = list(sys.path)
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "src_warcraft"))
    try:
        yield
    finally:
        sys.path[:] = saved_path
        for n in _COLLIDING:
            sys.modules.pop(n, None)
        sys.modules.update(saved_mods)


with _warcraft_first():
    from compute_metrics import _compute_one  # noqa: E402
    from model import is_valid_path  # noqa: E402
    from node_categories import (  # noqa: E402
        UnrepresentablePatternError,
        build_node_categories,
        edge_labels_to_node_labels,
        node_preds_to_edge_preds,
    )

GRID = 12
N_W = GRID - 1
N_RIGHT = GRID * N_W


def h_edge(i, j):
    return i * N_W + j


def v_edge(i, j):
    return N_RIGHT + i * GRID + j


# Detours up and left, so it is unrepresentable under the monotone alphabet.
NON_MONOTONE_PATH = (
    [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (0, 3)]
    + [(0, c) for c in range(4, 12)]
    + [(r, 11) for r in range(1, 12)]
)


def edge_vector(path):
    vec = np.zeros(2 * GRID * (GRID - 1), dtype=np.uint8)
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if r1 == r2:
            vec[h_edge(r1, min(c1, c2))] = 1
        else:
            vec[v_edge(min(r1, r2), c1)] = 1
    return vec


def edges_lookup():
    """The canonical (r1,c1,r2,c2) table, matching edges.npy."""
    rows = [None] * (2 * GRID * (GRID - 1))
    for i in range(GRID):
        for j in range(GRID - 1):
            rows[h_edge(i, j)] = (i, j, i, j + 1)
    for i in range(GRID - 1):
        for j in range(GRID):
            rows[v_edge(i, j)] = (i, j, i + 1, j)
    return np.array(rows)


def test_non_monotone_prediction_scores_exact_match_through_patched_metrics():
    """The gate: a correct non-monotone prediction must be credited EM=100."""
    gt = edge_vector(NON_MONOTONE_PATH)[None, :]
    pred = gt.copy()
    lookup = edges_lookup()
    vw = np.ones((1, GRID, GRID), dtype=np.float64)

    m = _compute_one(pred, gt, vw, lookup, GRID, apply_constraints=True)

    assert m["exact"] == 100.0, "non-monotone prediction was not credited EM"
    assert m["hamming"] == 100.0
    assert m["consistent"] == 100.0, "validity check rejected a valid path"
    assert m["ratio_eq_1.0"] == 100.0, "cost ratio mis-scored a correct path"
    assert m["n_valid"] == 1


def test_is_valid_path_accepts_the_non_monotone_route():
    """The consistency predicate must not encode a direction preference."""
    assert is_valid_path(edge_vector(NON_MONOTONE_PATH), edges_lookup(), GRID)


def test_full_alphabet_round_trips_the_prediction_the_scorer_sees():
    """The decode the scorer relies on must reproduce the path exactly."""
    vec = edge_vector(NON_MONOTONE_PATH)[None, :]
    ni, vr, no, total = build_node_categories(GRID, alphabet="full")
    labels = edge_labels_to_node_labels(vec, ni, no)

    oh = torch.zeros(1, total)
    off = 0
    for k, n in enumerate(vr):
        oh[0, off + labels[0, k]] = 1.0
        off += n
    decoded = (node_preds_to_edge_preds(oh, ni, no, vec.shape[1]) > 0.5).numpy()
    assert (decoded.astype(np.uint8) == vec).all()


def test_monotone_alphabet_still_cannot_score_it():
    """The contrast: the unpatched path would have mangled this prediction."""
    vec = edge_vector(NON_MONOTONE_PATH)[None, :]
    ni, _, no, _ = build_node_categories(GRID, alphabet="monotone")
    with pytest.raises(UnrepresentablePatternError):
        edge_labels_to_node_labels(vec, ni, no)


def test_metrics_cli_exposes_the_alphabet_and_mode():
    out = subprocess.run(
        [sys.executable, str(ROOT / "src_warcraft" / "compute_metrics.py"), "--help"],
        capture_output=True, text=True).stdout
    assert "--alphabet" in out
    assert "--constraint-mode" in out


def test_lazy_exact_requires_the_full_alphabet():
    """Guard: the restriction must not be re-imposed by a mismatched pairing."""
    r = subprocess.run(
        [sys.executable, str(ROOT / "src_warcraft" / "compute_metrics.py"),
         "--ckpt", "/nonexistent.pt", "--data_dir", "/nonexistent",
         "--grid-size", "12", "--constraint-mode", "lazy_exact",
         "--alphabet", "monotone"],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "requires --alphabet full" in (r.stderr + r.stdout)


def test_all_four_call_sites_accept_an_alphabet():
    for fname in ("compute_metrics.py", "compute_cost_ratio.py",
                  "cost_ratio.py", "train.py"):
        text = (ROOT / "src_warcraft" / fname).read_text()
        assert "alphabet=args.alphabet" in text, f"{fname} still hard-codes it"
        assert '"--alphabet"' in text, f"{fname} does not expose --alphabet"
