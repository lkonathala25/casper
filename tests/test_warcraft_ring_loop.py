import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "tasks/warcraft", ROOT / "tasks" / "warcraft", ROOT / "examples"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lazy_exact_layer import WarcraftLazyExactLayer  # noqa: E402

GRID = 12
N_W = GRID - 1
N_RIGHT = GRID * N_W


def h_edge(i, j):
    """Horizontal edge between (i,j) and (i,j+1)."""
    return i * N_W + j


def v_edge(i, j):
    """Vertical edge between (i,j) and (i+1,j)."""
    return N_RIGHT + i * GRID + j


def monotone_path_edges():
    """(0,0) -> right along row 0 -> down column 11 -> sink."""
    on = [h_edge(0, j) for j in range(GRID - 1)]
    on += [v_edge(i, GRID - 1) for i in range(GRID - 1)]
    return on


def unit_ring_edges(i, j):
    """The 4 edges of the unit square with top-left corner (i,j)."""
    return [h_edge(i, j), h_edge(i + 1, j), v_edge(i, j), v_edge(i, j + 1)]


@pytest.fixture(scope="module")
def layer():
    return WarcraftLazyExactLayer(GRID, alphabet="full", max_rounds=20)


def scores_from_on_edges(layer, on_edges, high=0.95, low=0.05):
    row = torch.full((layer.num_edges,), low)
    for e in on_edges:
        row[e] = high
    return row


def test_clean_path_certifies_in_one_round(layer):
    on = monotone_path_edges()
    assign, log = layer.project_board(scores_from_on_edges(layer, on))
    assert log["rounds"] == 1
    assert log["rings_per_round"] == [0]
    assert log["certified"]
    assert log["violations"] == []
    assert sorted(e for e, v in enumerate(assign) if v) == sorted(on)


def test_planted_ring_is_detected_and_eliminated(layer):
    """A path plus a disjoint ring must take >1 round and end ring-free."""
    on = monotone_path_edges() + unit_ring_edges(5, 3)
    row = scores_from_on_edges(layer, on)

    # The first projection should reproduce the planted configuration.
    first, _, _, _ = layer._project_one(row, overlay=None)
    walked, certified = layer.walk_certificate([e for e, v in enumerate(first) if v])
    assert not certified, "planted ring was not present in the first projection"
    rings = layer.find_rings([e for e, v in enumerate(first) if v], walked)
    assert len(rings) == 1
    assert sorted(rings[0]) == sorted(unit_ring_edges(5, 3))

    assign, log = layer.project_board(row)
    assert log["rounds"] >= 2, log
    assert log["rings_per_round"][0] == 1
    assert log["rings_per_round"][-1] == 0
    assert log["certified"]
    assert log["violations"] == []
    assert sum(log["edges_flipped_per_round"]) > 0

    on_final = [e for e, v in enumerate(assign) if v]
    _, cert = layer.walk_certificate(on_final)
    assert cert


def test_two_disjoint_rings_both_eliminated(layer):
    on = monotone_path_edges() + unit_ring_edges(5, 3) + unit_ring_edges(8, 6)
    assign, log = layer.project_board(scores_from_on_edges(layer, on))
    assert log["rings_per_round"][0] == 2
    assert log["certified"]
    assert log["violations"] == []
    _, cert = layer.walk_certificate([e for e, v in enumerate(assign) if v])
    assert cert


def test_validate_rejects_a_planted_ring(layer):
    """The independent check must catch a ring even without the walk."""
    row = [0] * layer.num_edges
    for e in monotone_path_edges() + unit_ring_edges(5, 3):
        row[e] = 1
    violations = layer.validate(row)
    assert violations, "independent validity check missed a stray ring"
    assert any("stranded" in v or "reachable" in v for v in violations)


def test_validate_rejects_degree_violations(layer):
    row = [0] * layer.num_edges
    for e in monotone_path_edges():
        row[e] = 1
    row[h_edge(3, 3)] = 1  # isolated edge: two degree-1 nodes
    violations = layer.validate(row)
    assert violations


def test_validate_accepts_a_clean_path(layer):
    row = [0] * layer.num_edges
    for e in monotone_path_edges():
        row[e] = 1
    assert layer.validate(row) == []


def test_forward_reports_per_board_logs(layer):
    """forward() must expose rounds, rings and flips for every board."""
    clean = monotone_path_edges()
    ringed = monotone_path_edges() + unit_ring_edges(5, 3)
    rows = torch.stack([scores_from_on_edges(layer, clean),
                        scores_from_on_edges(layer, ringed)])

    # Drive forward() through a one-hot categorical input matching those scores.
    labels = layer.bridge.edges_to_categories(
        torch.tensor([[1 if e in clean else 0 for e in range(layer.num_edges)],
                      [1 if e in ringed else 0 for e in range(layer.num_edges)]]))
    probs = torch.zeros(2, layer.total_dim)
    off = 0
    for k, vr in enumerate(layer.var_ranges):
        probs[torch.arange(2), off + labels[:, k]] = 1.0
        off += vr

    _, info = layer(probs)
    assert len(info["per_board"]) == 2
    assert info["per_board"][0]["rounds"] == 1
    assert info["per_board"][1]["rounds"] >= 2
    assert info["boards_needing_refinement"] == 1
    assert info["certified_boards"] == 2
    assert info["capped_boards"] == []
    assert info["empty_feasible_sets"] == 0
    assert rows.shape[0] == 2


def test_round_cap_is_respected(layer):
    """A tiny cap must surface as a capped board, not a silent pass."""
    tight = WarcraftLazyExactLayer(GRID, alphabet="full", max_rounds=1)
    on = monotone_path_edges() + unit_ring_edges(5, 3)
    _, log = tight.project_board(scores_from_on_edges(tight, on))
    assert log["rounds"] == 1
    assert log["capped"] is True
    assert log["certified"] is False
    assert tight.lifetime_capped_boards == 1
