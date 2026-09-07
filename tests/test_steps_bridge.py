import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("tasks/warcraft", "tasks/warcraft", "examples"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from _warcraft_import import load_warcraft  # noqa: E402

pytest.importorskip("torch")

import torch  # noqa: E402

from steps_bridge import PlainBroadcastBridge, softmax_per_node  # noqa: E402

N = 5


def test_source_and_sink_have_no_skip_category():
    br = PlainBroadcastBridge(N)
    assert br.skip_slots[0] == -1
    assert br.skip_slots[N * N - 1] == -1
    assert (br.skip_slots[1:-1] >= 0).all()


def test_interior_nodes_carry_seven_categories_under_the_full_alphabet():
    br = PlainBroadcastBridge(12)
    interior = br.node_info[(5, 5)]
    assert interior["n_cats"] == 7
    assert interior["categories"][0] == "skip"


def test_on_path_score_is_one_minus_p_skip():
    br = PlainBroadcastBridge(N)
    probs = torch.zeros(1, br.total_dim)
    off = 0
    for node in br.node_order:
        k = br.node_info[node]["n_cats"]
        probs[0, off:off + k] = 1.0 / k
        off += k
    scores = br.cell_scores(probs)
    for cell in range(N * N):
        slot = br.skip_slots[cell]
        if slot < 0:
            assert scores[0, cell] == 1.0
        else:
            assert np.isclose(scores[0, cell], 1.0 - float(probs[0, slot]))


def test_certain_skip_scores_zero_and_certain_on_path_scores_one():
    br = PlainBroadcastBridge(N)
    probs = torch.zeros(1, br.total_dim)
    off = 0
    for node in br.node_order:
        k = br.node_info[node]["n_cats"]
        probs[0, off] = 1.0  # slot 0 is 'skip' wherever a skip category exists
        off += k
    scores = br.cell_scores(probs)
    interior = [c for c in range(N * N) if br.skip_slots[c] >= 0]
    assert np.allclose(scores[0, interior], 0.0)
    assert scores[0, 0] == 1.0 and scores[0, N * N - 1] == 1.0


def test_tick_scores_are_broadcast_unchanged_across_time():
    br = PlainBroadcastBridge(N)
    cells = np.linspace(0.0, 1.0, N * N)
    ticks = br.tick_scores(cells, horizon=N * N - 1)
    assert ticks.shape == (N * N, N * N)
    for i in range(ticks.shape[0]):
        assert np.array_equal(ticks[i], cells)


def test_gt_cell_scores_mark_exactly_the_path_vertices():
    from eval_steps_bridge import edges_lookup_for

    br = PlainBroadcastBridge(N)
    lut = edges_lookup_for(N)
    n_w = N - 1
    vec = np.zeros((1, 2 * N * n_w), dtype=np.uint8)
    vec[0, 0] = 1  # (0,0)-(0,1)
    vec[0, 1] = 1  # (0,1)-(0,2)
    scores = br.gt_cell_scores(vec, lut)
    assert set(np.nonzero(scores[0])[0].tolist()) == {0, 1, 2}


def test_softmax_per_node_normalises_within_each_block():
    br = PlainBroadcastBridge(N)
    logits = torch.randn(3, br.total_dim)
    probs = softmax_per_node(logits, br.var_ranges)
    off = 0
    for vr in br.var_ranges:
        assert torch.allclose(
            probs[:, off:off + vr].sum(dim=1), torch.ones(3), atol=1e-5
        )
        off += vr


def test_bridge_applies_no_time_weighting():
    # Plain broadcast: every tick sees the same vector, so any two ticks agree.
    br = PlainBroadcastBridge(N)
    rng = np.random.default_rng(0)
    cells = rng.random(N * N)
    ticks = br.tick_scores(cells, horizon=10)
    assert np.array_equal(ticks[0], ticks[7])
