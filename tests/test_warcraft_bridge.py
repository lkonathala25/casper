import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "tasks" / "warcraft", ROOT / "casper"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from constraints import Constraints  # noqa: E402
from edge_bridge import CategoricalEdgeBridge, default_artifact  # noqa: E402
from warcraft_degree_constraints import node_incident  # noqa: E402

GRID = 12
N_TRIALS = 100


def sample_valid_edge_sets(grid_size, n, seed=0):
    """Random degree-legal edge assignments, drawn by walking the artifact."""
    cons = Constraints.from_compiled_dsl(default_artifact(grid_size))
    ordering = list(cons.ordering)
    n_edges = 2 * grid_size * (grid_size - 1)
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        assignment, ok = {}, True
        for step, var in enumerate(ordering):
            fs = sorted(cons.feasible_set(step, assignment))
            if not fs:
                ok = False
                break
            assignment[var] = rng.choice(fs) if len(fs) > 1 else fs[0]
        if ok:
            out.append([assignment[v] for v in range(n_edges)])
    return np.array(out, dtype=np.int64)


@pytest.fixture(scope="module")
def bridge():
    return CategoricalEdgeBridge(GRID, alphabet="full")


@pytest.fixture(scope="module")
def samples():
    return sample_valid_edge_sets(GRID, N_TRIALS, seed=0)


def test_samples_are_degree_legal(samples):
    """The generator itself must produce only degree-legal sets."""
    inc = node_incident(GRID)
    src, sink = (0, 0), (GRID - 1, GRID - 1)
    for row in samples:
        for node, edges in inc.items():
            deg = int(sum(row[e] for e in edges))
            if node in (src, sink):
                assert deg == 1, (node, deg)
            else:
                assert deg in (0, 2), (node, deg)


def test_edges_to_categories_to_edges_is_identity(samples, bridge):
    """edge -> category -> edge round trip on 100 random valid labelings."""
    labels = bridge.edges_to_categories(torch.from_numpy(samples))
    recovered = bridge.categories_to_edges(torch.from_numpy(labels)).numpy()
    assert recovered.shape == samples.shape
    mismatches = (recovered != samples).any(axis=1).sum()
    assert mismatches == 0, f"{mismatches}/{len(samples)} failed the round trip"


def test_categories_to_edges_to_categories_is_identity(samples, bridge):
    """category -> edge -> category round trip on the same 100 labelings."""
    labels = bridge.edges_to_categories(torch.from_numpy(samples))
    edges = bridge.categories_to_edges(torch.from_numpy(labels))
    again = bridge.edges_to_categories(edges)
    assert (again == labels).all()


def test_edge_scores_recover_a_onehot_labeling(samples, bridge):
    """A one-hot categorical input must score its own edges at exactly 1 / 0."""
    labels = bridge.edges_to_categories(torch.from_numpy(samples))
    B = len(samples)
    onehot = torch.zeros(B, bridge.total_dim)
    off = 0
    for k, vr in enumerate(bridge.var_ranges):
        onehot[torch.arange(B), off + labels[:, k]] = 1.0
        off += vr
    scores = bridge.edge_scores(onehot).numpy()
    assert np.allclose(scores, samples.astype(float), atol=1e-6)


def test_endpoint_marginals_agree_on_consistent_labelings(samples, bridge):
    """Both endpoints of every edge must report the same marginal."""
    labels = bridge.edges_to_categories(torch.from_numpy(samples))
    B = len(samples)
    onehot = torch.zeros(B, bridge.total_dim)
    off = 0
    for k, vr in enumerate(bridge.var_ranges):
        onehot[torch.arange(B), off + labels[:, k]] = 1.0
        off += vr
    mu, mv = bridge.endpoint_marginals(onehot)
    assert torch.allclose(mu, mv, atol=1e-6)


def test_ground_truth_paths_round_trip(bridge):
    """The real test-set labels must survive the bridge unchanged."""
    data = Path("/home/lk1325/mnist_experiment/warcraft_crl/"
                "warcraft_shortest_path_oneskin/12x12/preprocessed_4conn")
    if not data.exists():
        pytest.skip("dataset not available")
    gt = np.load(data / "test_edge_labels.npy").astype(np.int64)
    labels = bridge.edges_to_categories(torch.from_numpy(gt))
    recovered = bridge.categories_to_edges(torch.from_numpy(labels)).numpy()
    assert (recovered == gt).all()
