import sys
from itertools import product
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("tasks/warcraft", "casper"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from steps_encoding import (  # noqa: E402
    build_steps_encoding,
    manhattan,
    neighbours,
    to_dsl,
    trimmed_domain,
)

TEX_2X2_DOMAINS = [(0,), (1, 2), (3,), (3,)]

TEX_2X2_CLAUSES = [
    "[1,2]:X_1",
    "[2]:X_1 or [3]:X_2",
    "[1]:X_1 or [3]:X_2",
    "[3]:X_3",
]

TEX_2X2_MODELS = {(0, 1, 3, 3), (0, 2, 3, 3)}

TEX_3X3_DOMAINS = [
    (0,),
    (1, 3),
    (0, 2, 4, 6),
    (1, 3, 5, 7),
    (0, 2, 4, 6, 8),
    (1, 3, 5, 7, 8),
    (2, 4, 6, 8),
    (5, 7, 8),
    (8,),
]

TEX_3X3_MODELS = [
    (0, 1, 2, 5, 8, 8, 8, 8, 8),
    (0, 1, 2, 5, 4, 3, 6, 7, 8),
    (0, 1, 4, 3, 0, 3, 6, 7, 8),
]

# The four exclusions listed in the tex, in the order it lists them. The first
# and third are domain-legal but break a transition clause; the second and
# fourth are ruled out by domain membership alone.
TEX_3X3_EXCLUSIONS = [
    ("leaves the target after arriving", (0, 1, 2, 5, 8, 5, 8, 8, 8)),
    ("off its parity class (x_2 = 1)", (0, 1, 1, 3, 4, 5, 8, 8, 8)),
    ("jump between non-adjacent cells", (0, 1, 6, 7, 8, 8, 8, 8, 8)),
    ("strays too far to return (x_6 = 0)", (0, 1, 4, 3, 0, 3, 0, 7, 8)),
]


def absorbing_walks(g):
    """Every absorbing s-t walk of length L, enumerated directly from the grid."""
    n = g * g
    L = n - 1
    t = n - 1
    out = []

    def rec(prefix):
        if len(prefix) == L + 1:
            if prefix[-1] == t:
                out.append(tuple(prefix))
            return
        u = prefix[-1]
        for v in ((t,) if u == t else neighbours(g, u)):
            prefix.append(v)
            rec(prefix)
            prefix.pop()

    rec([0])
    return out


def simple_paths(g):
    """Every simple s-t path on the grid, as a cell sequence."""
    n = g * g
    t = n - 1
    out = []

    def rec(prefix, seen):
        u = prefix[-1]
        if u == t:
            out.append(tuple(prefix))
            return
        for v in neighbours(g, u):
            if v in seen:
                continue
            prefix.append(v)
            seen.add(v)
            rec(prefix, seen)
            seen.remove(v)
            prefix.pop()

    rec([0], {0})
    return out


def test_2x2_domains_match_tex():
    enc = build_steps_encoding(2)
    assert list(enc.domains) == TEX_2X2_DOMAINS


def test_2x2_clauses_match_tex_exactly():
    enc = build_steps_encoding(2)
    assert [repr(c) for c in enc.clauses] == TEX_2X2_CLAUSES


def test_2x2_has_exactly_the_two_tex_models():
    enc = build_steps_encoding(2)
    assert set(enc.models()) == TEX_2X2_MODELS


def test_3x3_domains_match_tex_table():
    enc = build_steps_encoding(3)
    assert list(enc.domains) == TEX_3X3_DOMAINS


def test_3x3_domain_table_holds_29_values_against_81_untrimmed():
    enc = build_steps_encoding(3)
    assert sum(len(d) for d in enc.domains) == 29
    # The tex's "81 untrimmed" is the full vertex set at each of the 9 steps.
    assert (enc.horizon + 1) * enc.n == 81


@pytest.mark.parametrize("model", TEX_3X3_MODELS)
def test_3x3_tex_sample_models_satisfy(model):
    enc = build_steps_encoding(3)
    assert enc.satisfies(model)


@pytest.mark.parametrize("label,assignment", TEX_3X3_EXCLUSIONS)
def test_3x3_tex_exclusions_violate(label, assignment):
    enc = build_steps_encoding(3)
    assert not enc.satisfies(assignment), label


def test_3x3_parity_exclusion_is_a_domain_fact():
    enc = build_steps_encoding(3)
    assert 1 not in enc.domains[2]


def test_3x3_backward_exclusion_is_a_domain_fact():
    enc = build_steps_encoding(3)
    assert 0 not in enc.domains[6]
    assert manhattan(3, 0, 8) > enc.horizon - 6


def test_3x3_loop_erasure_of_the_cyclic_sample_model():
    # The tex states the cyclic model erases to (0, 3, 6, 7, 8).
    walk = TEX_3X3_MODELS[2]
    erased = []
    for v in walk:
        if v in erased:
            del erased[erased.index(v) + 1:]
        else:
            erased.append(v)
    assert tuple(erased) == (0, 3, 6, 7, 8)


@pytest.mark.parametrize("g", [2, 3])
def test_models_are_exactly_the_absorbing_walks(g):
    enc = build_steps_encoding(g)
    assert set(enc.models()) == set(absorbing_walks(g))


@pytest.mark.parametrize("g", [2, 3, 4, 5])
def test_every_simple_path_is_a_model_when_padded(g):
    enc = build_steps_encoding(g)
    L = enc.horizon
    paths = simple_paths(g)
    assert paths
    for p in paths:
        padded = p + (enc.target,) * (L + 1 - len(p))
        assert enc.satisfies(padded), p


@pytest.mark.parametrize("g", [2, 3, 4, 5, 12])
def test_no_reachable_state_can_dead_end(g):
    enc = build_steps_encoding(g)
    for c in enc.clauses:
        assert c.succ, f"dead end at step {c.step}"


@pytest.mark.parametrize("g", [3, 4, 5, 12])
def test_endpoints_are_domain_facts(g):
    enc = build_steps_encoding(g)
    assert enc.domains[0] == (enc.source,)
    assert enc.domains[enc.horizon] == (enc.target,)


@pytest.mark.parametrize("g", [3, 4, 5, 12])
def test_target_enters_only_from_its_distance_onwards(g):
    enc = build_steps_encoding(g)
    d_st = manhattan(g, enc.source, enc.target)
    for i, d in enumerate(enc.domains):
        assert (enc.target in d) == (i >= d_st), i


@pytest.mark.parametrize("g", [3, 4, 5, 12])
def test_clause_count_is_the_sum_of_domain_sizes_before_the_horizon(g):
    enc = build_steps_encoding(g)
    assert len(enc.clauses) == sum(len(enc.domains[i]) for i in range(enc.horizon))


def test_no_alldifferent_clauses_are_emitted():
    # Every clause touches exactly two consecutive step variables, so the primal
    # graph is the chain X_0 - ... - X_L of induced width 1.
    enc = build_steps_encoding(4)
    steps = {c.step for c in enc.clauses}
    assert steps == set(range(enc.horizon))
    assert len(enc.clauses) == sum(len(enc.domains[i]) for i in range(enc.horizon))


def test_guard_is_dropped_exactly_on_singleton_domains():
    enc = build_steps_encoding(4)
    for c in enc.clauses:
        singleton = len(enc.domains[c.step]) == 1
        assert (c.guard == ()) == singleton, c


def test_dsl_round_trips_through_the_parser():
    from signed_clauses import parse_constraints

    enc = build_steps_encoding(3)
    cs = parse_constraints(to_dsl(enc))
    assert cs.var_names == [f"X_{i}" for i in range(enc.horizon + 1)]
    assert cs.var_domains == [len(d) for d in enc.domains]
    assert len(cs.clauses) == len(enc.clauses)

    # Local values decode back to the tex's cell indices.
    for parsed, built in zip(cs.clauses, enc.clauses):
        lits = {lit.var: lit.values for lit in parsed.literals}
        succ = {enc.domains[built.step + 1][k] for k in lits[built.step + 1]}
        assert succ == set(built.succ)
        if built.guard:
            guard = {enc.domains[built.step][k] for k in lits[built.step]}
            assert guard == set(built.guard)


def test_dsl_models_agree_with_the_encoding_models():
    from signed_clauses import constraints_satisfied, parse_constraints

    enc = build_steps_encoding(3)
    cs = parse_constraints(to_dsl(enc))
    found = set()
    for local in product(*(range(k) for k in cs.var_domains)):
        if constraints_satisfied(cs.clauses, dict(enumerate(local))):
            found.add(tuple(enc.domains[i][v] for i, v in enumerate(local)))
    assert found == set(enc.models())


def test_12x12_trimmed_matches_the_tex_compile_table_clause_count():
    from resolution import _is_tautology
    from signed_clauses import parse_constraints

    enc = build_steps_encoding(12)
    assert len(enc.clauses) == 8844
    cs = parse_constraints(to_dsl(enc))
    kept = [
        c
        for c in cs.clauses
        if c.literals and not _is_tautology(c, cs.var_domains)
    ]
    assert len(kept) == 8839


def test_30x30_trimmed_matches_the_tex_compile_table_clause_count():
    enc = build_steps_encoding(30)
    # 379,315 in the tex, plus the 5 tautologies COMPILE drops before counting.
    assert len(enc.clauses) == 379320


def test_trimmed_domain_helper_agrees_with_the_builder():
    enc = build_steps_encoding(5)
    for i in range(enc.horizon + 1):
        assert trimmed_domain(5, i) == enc.domains[i]
