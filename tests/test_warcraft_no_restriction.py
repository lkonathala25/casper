import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src_warcraft", ROOT / "src", ROOT / "examples"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from edge_bridge import CategoricalEdgeBridge  # noqa: E402
from lazy_exact_layer import (  # noqa: E402
    WarcraftEdgeConstraintLayer,
    build_constraint_layer,
)

GRID = 12


def edge_vector(path, N):
    n_w = N - 1
    n_right = N * n_w
    vec = np.zeros(2 * N * (N - 1), dtype=np.int64)
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if r1 == r2:
            vec[r1 * n_w + min(c1, c2)] = 1
        else:
            vec[n_right + min(r1, r2) * N + c1] = 1
    return vec


def decode_path(edge_vec, N):
    """Ordered node sequence from a binary edge vector, or None if not a path."""
    n_w = N - 1
    n_right = N * n_w
    adj = {}
    for idx in np.nonzero(np.asarray(edge_vec) > 0.5)[0]:
        if idx < n_right:
            r, c = divmod(int(idx), n_w)
            a, b = (r, c), (r, c + 1)
        else:
            r, c = divmod(int(idx) - n_right, N)
            a, b = (r, c), (r + 1, c)
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    src, dst = (0, 0), (N - 1, N - 1)
    if src not in adj:
        return None
    seq, prev, cur = [src], None, src
    for _ in range(2 * N * N):
        nxt = [x for x in adj[cur] if x != prev]
        if not nxt:
            return None
        prev, cur = cur, nxt[0]
        seq.append(cur)
        if cur == dst:
            return seq
    return None


def is_monotone(path):
    return all(b[0] >= a[0] and b[1] >= a[1] for a, b in zip(path, path[1:]))


# Detours up and left out of the top-left corner, then runs to the sink.
NON_MONOTONE_PATH = (
    [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (0, 3)]
    + [(0, c) for c in range(4, 12)]
    + [(r, 11) for r in range(1, 12)]
)


def onehot_from_labels(labels, var_ranges, total_dim):
    B = labels.shape[0]
    oh = torch.zeros(B, total_dim)
    off = 0
    for k, vr in enumerate(var_ranges):
        oh[torch.arange(B), off + labels[:, k]] = 1.0
        off += vr
    return oh


@pytest.fixture(scope="module")
def layer():
    return WarcraftEdgeConstraintLayer(GRID, alphabet="full")


def test_fixture_path_is_valid_and_non_monotone():
    p = NON_MONOTONE_PATH
    assert p[0] == (0, 0) and p[-1] == (GRID - 1, GRID - 1)
    assert len(set(p)) == len(p), "fixture path must be simple"
    assert not is_monotone(p)
    assert decode_path(edge_vector(p, GRID), GRID) == p


def test_valid_non_monotone_path_passes_layer_unedited(layer):
    """(a) A known valid non-monotone path must survive the layer untouched."""
    vec = edge_vector(NON_MONOTONE_PATH, GRID)[None, :]
    labels = layer.bridge.edges_to_categories(torch.from_numpy(vec))
    probs = onehot_from_labels(torch.from_numpy(labels), layer.var_ranges,
                               layer.total_dim)

    adjusted, info = layer(probs)

    assert info["empty_feasible_sets"] == 0
    assert info["cells_edited"] == 0, "layer edited a already-valid path"
    assert (info["edge_assignment"].numpy() == vec).all()
    # argmax per node is unchanged
    off = 0
    for k, vr in enumerate(layer.var_ranges):
        assert int(adjusted[0, off:off + vr].argmax()) == int(labels[0, k])
        off += vr


def test_layer_can_output_a_non_monotone_path(layer):
    """(b) With scores favouring an up/left route, the layer must produce one."""
    vec = edge_vector(NON_MONOTONE_PATH, GRID)[None, :]
    labels = layer.bridge.edges_to_categories(torch.from_numpy(vec))

    # Soft, deliberately non-one-hot: 0.6 on the favoured category, rest spread.
    probs = torch.zeros(1, layer.total_dim)
    off = 0
    for k, vr in enumerate(layer.var_ranges):
        probs[0, off:off + vr] = 0.4 / (vr - 1) if vr > 1 else 1.0
        probs[0, off + int(labels[0, k])] = 0.6
        off += vr

    _, info = layer(probs)
    out_path = decode_path(info["edge_assignment"][0].numpy(), GRID)

    assert out_path is not None, "layer output did not decode to a path"
    assert not is_monotone(out_path), (
        "layer produced a monotone path even though scores favoured an "
        "up/left route — the search space is still restricted")
    assert info["empty_feasible_sets"] == 0


def test_monotone_layer_cannot_represent_the_same_path():
    """The old mode is still available and still restricted — the contrast."""
    from node_categories import (
        UnrepresentablePatternError,
        build_node_categories,
        edge_labels_to_node_labels,
    )
    vec = edge_vector(NON_MONOTONE_PATH, GRID)[None, :]
    ni, _, no, _ = build_node_categories(GRID, alphabet="monotone")
    with pytest.raises(UnrepresentablePatternError):
        edge_labels_to_node_labels(vec, ni, no)

    mono = build_constraint_layer("monotone", grid_size=GRID)
    assert type(mono).__name__ == "WarcraftConstraintLayer"


def test_no_directional_masking_in_the_new_path():
    """(c) Audit: the lazy-exact path must not reference the monotone sweep."""
    banned = [
        r"flow_from_R", r"flow_from_D",
        r"_emitting_[RD]_sets", r"_accepting_[RD]_sets",
        r"_left_idx", r"_above_idx",
        r"processing_order",
    ]
    for fname in ("lazy_exact_layer.py", "edge_bridge.py"):
        text = (ROOT / "src_warcraft" / fname).read_text()
        # The factory names the old mode; that mention is allowed.
        text = text.replace("processing_order=processing_order", "")
        text = text.replace("processing_order='row'", "")
        for pat in banned:
            hits = [ln for ln in text.splitlines() if re.search(pat, ln)]
            assert not hits, f"{fname} still references {pat}: {hits}"


def test_full_alphabet_admits_all_four_directions():
    """Every interior node must carry both backward-corner categories."""
    bridge = CategoricalEdgeBridge(GRID, alphabet="full")
    for i in range(1, GRID - 1):
        for j in range(1, GRID - 1):
            cats = bridge.node_info[(i, j)]["categories"]
            assert "L|U" in cats and "R|D" in cats, (i, j, cats)


def test_grep_audit_reports_monotone_confined_to_legacy_module():
    """The directional sweep must live only in the legacy layer module."""
    out = subprocess.run(
        ["grep", "-rln", "--include=*.py", "flow_from_R",
         str(ROOT / "src_warcraft")],
        capture_output=True, text=True).stdout.split()
    assert [Path(p).name for p in out] == ["constraint_layer.py"], out
