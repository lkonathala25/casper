import sys
from pathlib import Path

import numpy as np
import pytest

SRC_WARCRAFT = Path(__file__).resolve().parents[1] / "tasks" / "warcraft"
if str(SRC_WARCRAFT) not in sys.path:
    sys.path.insert(0, str(SRC_WARCRAFT))

from node_categories import (  # noqa: E402
    UnrepresentablePatternError,
    build_node_categories,
    edge_labels_to_node_labels,
    node_preds_to_edge_preds,
)

N = 5


def edge_vector(path, N):
    """Binary edge vector for an ordered node path, using the canonical indexing."""
    n_w = N - 1
    n_right = N * n_w
    vec = np.zeros(2 * N * (N - 1), dtype=int)
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if r1 == r2:
            vec[r1 * n_w + min(c1, c2)] = 1
        else:
            vec[n_right + min(r1, r2) * N + c1] = 1
    return vec


MONOTONE_PATH = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
                 (1, 4), (2, 4), (3, 4), (4, 4)]

# Detours up and left, so it uses both {left,above} and {right,below} corners.
NON_MONOTONE_PATH = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (0, 3),
                     (1, 3), (1, 2), (2, 2), (2, 3), (2, 4), (3, 4), (4, 4)]


def test_monotone_alphabet_matches_recorded_sizes():
    _, _, _, td = build_node_categories(12, alphabet="monotone")
    assert td == 628
    _, _, _, td = build_node_categories(12, alphabet="full")
    assert td == 868


def test_full_alphabet_is_prefix_extension_of_monotone():
    mi, _, order, _ = build_node_categories(N, alphabet="monotone")
    fi, _, _, _ = build_node_categories(N, alphabet="full")
    for node in order:
        m, f = mi[node], fi[node]
        assert f["edge_patterns"][: m["n_cats"]] == m["edge_patterns"], node


def test_monotone_path_converts_under_both_alphabets():
    vec = edge_vector(MONOTONE_PATH, N)[None, :]
    for alphabet in ("monotone", "full"):
        info, _, order, _ = build_node_categories(N, alphabet=alphabet)
        edge_labels_to_node_labels(vec, info, order)


def test_non_monotone_path_raises_under_monotone_alphabet():
    vec = edge_vector(NON_MONOTONE_PATH, N)[None, :]
    info, _, order, _ = build_node_categories(N, alphabet="monotone")
    with pytest.raises(UnrepresentablePatternError) as exc:
        edge_labels_to_node_labels(vec, info, order, instance_ids=["inst-42"])
    msg = str(exc.value)
    assert "inst-42" in msg
    assert "no category" in msg
    # The message must name the offending node and its on-edges.
    assert "node (" in msg
    assert "on-edges" in msg


def test_non_monotone_path_round_trips_under_full_alphabet():
    import torch

    vec = edge_vector(NON_MONOTONE_PATH, N)
    info, var_ranges, order, _ = build_node_categories(N, alphabet="full")
    labels = edge_labels_to_node_labels(vec[None, :], info, order)

    oh = torch.zeros(1, sum(var_ranges))
    off = 0
    for k, vr in enumerate(var_ranges):
        oh[0, off + labels[0, k]] = 1.0
        off += vr
    recovered = (node_preds_to_edge_preds(oh, info, order, len(vec)) > 0.5).numpy().astype(int)
    assert (recovered[0] == vec).all()


def test_no_silent_fallback_remains():
    """A pattern with three incident edges on must raise, never relabel to skip."""
    info, _, order, _ = build_node_categories(N, alphabet="full")
    vec = np.zeros(2 * N * (N - 1), dtype=int)
    n_w = N - 1
    n_right = N * n_w
    # Node (2,2): left, right and below edges all on -> degree 3, unrepresentable.
    vec[2 * n_w + 1] = 1
    vec[2 * n_w + 2] = 1
    vec[n_right + 2 * N + 2] = 1
    with pytest.raises(UnrepresentablePatternError):
        edge_labels_to_node_labels(vec[None, :], info, order)
