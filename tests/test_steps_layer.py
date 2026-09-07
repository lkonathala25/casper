import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("tasks/warcraft", "tasks/warcraft"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from _warcraft_import import load_warcraft  # noqa: E402

pytest.importorskip("torch")

import torch  # noqa: E402

_nc = load_warcraft("node_categories")
build_node_categories = _nc.build_node_categories
edge_labels_to_node_labels = _nc.edge_labels_to_node_labels

from steps_layer import StepsProjectionLayer, one_hot_from_categories  # noqa: E402
from steps_walk import sequence_to_edges  # noqa: E402

G = 5
PATH = [0, 5, 6, 11, 16, 17, 18, 23, 24]


@pytest.fixture(scope="module")
def layer():
    return StepsProjectionLayer(G, alphabet="full")


@pytest.fixture(scope="module")
def cats():
    node_info, var_ranges, node_order, _ = build_node_categories(G, alphabet="full")
    vec = sequence_to_edges(G, PATH)[None, :]
    return edge_labels_to_node_labels(vec, node_info, node_order), var_ranges


def node_argmax(proj, var_ranges):
    """Per-node argmax category of a (B, total_dim) distribution."""
    out = np.zeros((proj.shape[0], len(var_ranges)), dtype=np.int64)
    off = 0
    for k, vr in enumerate(var_ranges):
        out[:, k] = proj[:, off:off + vr].argmax(dim=1).numpy()
        off += vr
    return out


def test_round_trip_recovers_the_path_categories(layer, cats):
    gt_cats, var_ranges = cats
    proj, _ = layer(one_hot_from_categories(gt_cats, var_ranges))
    assert (node_argmax(proj, var_ranges) == gt_cats).all()


def test_target_categories_match_the_decoded_edge_set(layer, cats):
    gt_cats, var_ranges = cats
    _, info = layer(one_hot_from_categories(gt_cats, var_ranges))
    assert np.array_equal(info["target_categories"].numpy(), gt_cats)
    assert np.array_equal(
        info["edge_assignment"].numpy()[0].astype(bool),
        sequence_to_edges(G, PATH).astype(bool),
    )


def test_output_is_not_one_hot(layer):
    # The reverse bridge min-edits onto the target rather than hard-relabelling.
    probs = torch.softmax(torch.randn(4, layer.total_dim) * 0.5, dim=1)
    proj, _ = layer(probs)
    assert proj.max() < 1.0
    assert (proj > 1e-6).sum() > proj.shape[0] * len(layer.var_ranges)


def test_mass_is_conserved_in_every_node_block(layer):
    probs = torch.zeros(3, layer.total_dim)
    off = 0
    for vr in layer.var_ranges:
        probs[:, off:off + vr] = torch.softmax(torch.randn(3, vr), dim=1)
        off += vr
    proj, _ = layer(probs)
    off = 0
    for vr in layer.var_ranges:
        assert torch.allclose(
            proj[:, off:off + vr].sum(dim=1), torch.ones(3), atol=1e-5
        )
        off += vr


def test_untouched_categories_keep_their_probability(layer):
    # min-edit only moves the target and its active set; everything else is
    # left exactly as it came in.
    probs = torch.zeros(1, layer.total_dim)
    off = 0
    for vr in layer.var_ranges:
        probs[0, off:off + vr] = torch.softmax(torch.randn(vr) * 3, dim=0)
        off += vr
    proj, _ = layer(probs)
    assert (proj == probs).any()


def test_projection_is_deterministic(layer):
    probs = torch.softmax(torch.randn(6, layer.total_dim), dim=1)
    a, _ = layer(probs)
    b, _ = layer(probs)
    assert torch.equal(a, b)


def test_no_dead_ends_and_shape_preserved(layer):
    probs = torch.softmax(torch.randn(8, layer.total_dim), dim=1)
    proj, info = layer(probs)
    assert proj.shape == probs.shape
    assert info["dead_ends"] == 0


def test_certify_flag_reports_certificates(layer_certifying=None):
    lay = StepsProjectionLayer(G, alphabet="full", certify=True)
    probs = torch.softmax(torch.randn(8, lay.total_dim), dim=1)
    _, info = lay(probs)
    assert info["certified_boards"] == 8
    assert info["uncertified_boards"] == []
