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

_is_valid_src_sink_path = load_warcraft(
    "compute_cost_ratio")._is_valid_src_sink_path

from steps_walk import (  # noqa: E402
    StepsWalkDecoder,
    edge_endpoints_array,
    loop_erase,
    sequence_to_edges,
)

TEX_CYCLIC_MODEL = (0, 1, 4, 3, 0, 3, 6, 7, 8)


def certificate(g, cells):
    """Run the edge-based pipeline's own validity check on a decoded cell path."""
    vec = sequence_to_edges(g, cells).astype(bool)
    return _is_valid_src_sink_path(
        vec, edge_endpoints_array(g), 0, (g - 1) * g + (g - 1)
    )


def test_loop_erase_matches_the_tex_worked_example():
    erased, snipped = loop_erase(TEX_CYCLIC_MODEL)
    assert tuple(erased) == (0, 3, 6, 7, 8)
    assert snipped == len(TEX_CYCLIC_MODEL) - 5


def test_loop_erase_is_idempotent():
    erased, _ = loop_erase(TEX_CYCLIC_MODEL)
    again, snipped = loop_erase(erased)
    assert again == erased
    assert snipped == 0


def test_loop_erase_leaves_a_simple_path_untouched():
    path = (0, 1, 2, 5, 8)
    erased, snipped = loop_erase(path)
    assert tuple(erased) == path
    assert snipped == 0


def test_erased_tex_model_passes_the_certificate():
    erased, _ = loop_erase(TEX_CYCLIC_MODEL)
    assert certificate(3, erased)


def test_unerased_cyclic_walk_fails_the_certificate():
    # The certificate is a real gate, not a formality: the raw cyclic walk
    # revisits cells and must not pass it.
    assert not certificate(3, TEX_CYCLIC_MODEL)


def test_decode_trims_the_absorbing_padding():
    dec = StepsWalkDecoder(3)
    padded = (0, 1, 2, 5, 8, 8, 8, 8, 8)
    erased, snipped, vec = dec.decode(padded)
    assert tuple(erased) == (0, 1, 2, 5, 8)
    assert snipped == 0
    assert vec.sum() == 4


@pytest.mark.parametrize("g", [3, 4, 5])
def test_walk_never_dead_ends_and_always_certifies(g):
    dec = StepsWalkDecoder(g)
    rng = np.random.default_rng(7)
    for _ in range(20):
        scores = rng.normal(size=(dec.enc.horizon + 1, g * g))
        seq, stats = dec.walk(scores)
        assert stats.dead_ends == 0
        erased, _, vec = dec.decode(seq)
        assert certificate(g, erased)


@pytest.mark.parametrize("g", [3, 4, 5])
def test_early_cut_changes_no_output(g):
    dec = StepsWalkDecoder(g)
    rng = np.random.default_rng(11)
    for _ in range(20):
        scores = rng.normal(size=(dec.enc.horizon + 1, g * g))
        cut, _ = dec.walk(scores, early_cut=True)
        full, _ = dec.walk(scores, early_cut=False)
        assert cut == full


@pytest.mark.parametrize("g", [3, 4, 5])
def test_walk_output_is_a_model_of_the_encoding(g):
    dec = StepsWalkDecoder(g)
    rng = np.random.default_rng(13)
    for _ in range(20):
        scores = rng.normal(size=(dec.enc.horizon + 1, g * g))
        seq, _ = dec.walk(scores)
        assert dec.enc.satisfies(seq)


def test_feasible_set_is_succ_intersect_domain():
    dec = StepsWalkDecoder(4)
    for i in range(1, dec.enc.horizon + 1):
        for u in dec.enc.domains[i - 1]:
            f = dec.feasible(i, u)
            assert set(f) <= set(dec.enc.domains[i])
            assert f, (i, u)
            assert len(f) <= 4


def test_absorption_leaves_exactly_one_successor():
    dec = StepsWalkDecoder(4)
    t = dec.enc.target
    d_st = 2 * (4 - 1)
    for i in range(d_st + 1, dec.enc.horizon + 1):
        assert dec.feasible(i, t) == (t,)


def test_sequence_to_edges_uses_the_canonical_indexing():
    # Row 0 rightward edges come first, then verticals: (0,0)-(0,1) is index 0.
    vec = sequence_to_edges(3, [0, 1])
    assert vec.sum() == 1 and vec[0] == 1
    _, n_w = 3, 2
    vec = sequence_to_edges(3, [0, 3])
    assert vec.sum() == 1 and vec[3 * n_w] == 1


def test_tie_break_prefers_a_non_reversal_among_tied_candidates():
    dec = StepsWalkDecoder(4)
    cands = (1, 4, 9)
    vals = [0.5, 0.5, 0.2]
    assert dec._break_tie(cands, vals, 0.5, "no-backtrack", reversal=1) == 4
    assert dec._break_tie(cands, vals, 0.5, "index", reversal=1) == 1


def test_tie_break_leaves_a_strict_winner_alone_even_if_it_reverses():
    # The adopted rule is a tie-break, not a ban on backward moves.
    dec = StepsWalkDecoder(4)
    cands = (1, 4, 9)
    vals = [0.9, 0.5, 0.2]
    assert dec._break_tie(cands, vals, 0.9, "no-backtrack", reversal=1) == 1


def test_tie_break_keeps_the_reversal_when_it_is_the_only_tied_candidate():
    dec = StepsWalkDecoder(4)
    assert dec._break_tie((1,), [0.7], 0.7, "no-backtrack", reversal=1) == 1


def test_gt_style_scores_recover_the_path_without_erasure():
    # Exactly the pass-through regime: 1.0 on-path, 0.0 off. With the adopted
    # tie-break the walk marches the path instead of stalling on it.
    g = 5
    dec = StepsWalkDecoder(g)
    path = [0, 5, 6, 11, 16, 17, 18, 23, 24]
    scores = np.zeros((dec.enc.horizon + 1, g * g))
    scores[:, path] = 1.0
    seq, stats = dec.walk(scores)
    erased, snipped, _ = dec.decode(seq)
    assert erased == path
    assert snipped == 0
    assert stats.dead_ends == 0
