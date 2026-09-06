from itertools import product
from pathlib import Path

import pytest

from signed_clauses import (
    SignedClause,
    SignedLiteral,
    parse_constraints,
)
from resolution import (
    ClauseFeasibility,
    CompiledConstraints,
    Unsatisfiable,
    compile_constraints,
    enumerate_models,
    from_json,
    to_json,
)


EXAMPLE1_PATH = Path(__file__).resolve().parents[1] / "examples" / "example1_shapes.txt"
EXAMPLE1_TEXT = EXAMPLE1_PATH.read_text()


def _lit(var: int, *values: int) -> SignedLiteral:
    return SignedLiteral(var, frozenset(values))


def _clause(*lits: SignedLiteral) -> SignedClause:
    return SignedClause.from_literals(lits)


def test_parser_basic_example1():
    cs = parse_constraints(EXAMPLE1_TEXT)
    assert cs.var_names == ["y_1", "y_2", "y_3"]
    assert cs.var_domains == [3, 3, 2]
    assert cs.ordering == [0, 1, 2]
    assert len(cs.clauses) == 2

    c1, c2 = cs.clauses
    assert c1 == _clause(_lit(1, 0), _lit(2, 0))
    assert c2 == _clause(_lit(0, 1, 2), _lit(1, 0))


def test_parser_merges_duplicate_vars_in_clause():
    """A clause `[0]:y_1 or [1]:y_1` folds to `[0,1]:y_1`."""
    text = "var y_1 3\n[0]:y_1 or [1]:y_1"
    cs = parse_constraints(text)
    assert cs.clauses == [_clause(_lit(0, 0, 1))]


def test_parser_explicit_ordering():
    text = """
    var a 2
    var b 2
    ordering b a
    [0]:a
    """
    cs = parse_constraints(text)
    assert cs.ordering == [1, 0]


def test_parser_errors():
    from signed_clauses import ParseError

    with pytest.raises(ParseError):
        parse_constraints("var y_1 3\n[7]:y_1")

    with pytest.raises(ParseError):
        parse_constraints("[0]:y_1")

    with pytest.raises(ParseError):
        parse_constraints("var y_1 3\nvar y_1 5")


def test_compile_example1_matches_hand_derivation():
    """Inference order y_1, y_2, y_3 -> COMPILE processes i=3, 2, 1."""
    cs = parse_constraints(EXAMPLE1_TEXT)
    cc = compile_constraints(cs)

    c1 = _clause(_lit(1, 0), _lit(2, 0))
    c2 = _clause(_lit(0, 1, 2), _lit(1, 0))

    assert cc.per_var_clauses[0] == []
    assert cc.per_var_clauses[1] == [c2]
    assert cc.per_var_clauses[2] == [c1]


def test_compile_unsat_two_contradictory_units():
    """[0]:y_1 and [1]:y_1 force y_1 into ∅; empty clause derived."""
    text = """
    var y_1 2
    [0]:y_1
    [1]:y_1
    """
    cs = parse_constraints(text)
    with pytest.raises(Unsatisfiable):
        compile_constraints(cs)


def test_compile_unsat_via_conflict_propagation():
    """A two-variable inconsistency that surfaces only after one resolution step."""
    text = """
    var y_1 2
    var y_2 2
    [0]:y_1 or [0]:y_2
    [1]:y_1 or [0]:y_2
    [0]:y_1 or [1]:y_2
    [1]:y_1 or [1]:y_2
    """
    cs = parse_constraints(text)
    with pytest.raises(Unsatisfiable):
        compile_constraints(cs)


def test_compile_drops_input_tautologies():
    """A clause with a full-domain literal is satisfied by every assignment."""
    text = """
    var y_1 3
    var y_2 2
    [0,1,2]:y_1 or [0]:y_2
    [0]:y_2
    """
    cs = parse_constraints(text)
    cc = compile_constraints(cs)
    assert cc.per_var_clauses[1] == [_clause(_lit(1, 0))]
    assert cc.per_var_clauses[0] == []


def test_compile_drops_merged_literal_tautologies():
    """When merged φ_j literals cover the full domain of some variable, drop."""
    text = """
    var y_1 2
    var y_2 2
    [0]:y_2 or [0]:y_1
    [1]:y_2 or [1]:y_1
    """
    cs = parse_constraints(text)
    cc = compile_constraints(cs)
    assert cc.per_var_clauses[0] == []


def _models_of_compiled(cc: CompiledConstraints) -> set[tuple[int, ...]]:
    all_clauses = [c for pi_bar in cc.per_var_clauses for c in pi_bar]
    return set(enumerate_models(all_clauses, cc.var_domains))


def _models_of_input(cs) -> set[tuple[int, ...]]:
    return set(enumerate_models(cs.clauses, cs.var_domains))


def test_lemma2_example1_brute_force():
    cs = parse_constraints(EXAMPLE1_TEXT)
    cc = compile_constraints(cs)
    assert _models_of_input(cs) == _models_of_compiled(cc)


@pytest.mark.parametrize("seed", range(8))
def test_lemma2_random_small_brute_force(seed: int):
    """Random small Π over 3 vars × 3 values: compile preserves the model set."""
    import random

    from resolution import CompileTooLarge

    rng = random.Random(seed)
    text_lines = ["var a 3", "var b 3", "var c 3"]
    for _ in range(5):
        k = rng.randint(1, 3)
        chosen_vars = rng.sample(["a", "b", "c"], k)
        lits = []
        for v in chosen_vars:
            n_vals = rng.randint(1, 2)
            vals = rng.sample([0, 1, 2], n_vals)
            lits.append("[" + ",".join(str(x) for x in vals) + "]:" + v)
        text_lines.append(" or ".join(lits))
    text = "\n".join(text_lines)
    cs = parse_constraints(text)

    input_models = _models_of_input(cs)
    if not input_models:
        with pytest.raises(Unsatisfiable):
            compile_constraints(cs)
        return

    try:
        cc = compile_constraints(cs)
    except CompileTooLarge:
        pytest.skip("random Π too dense for literal Eq. (11) compiler")

    compiled_models = _models_of_compiled(cc)
    assert input_models == compiled_models, (
        f"seed={seed}: model set diverged.\n"
        f"  input Π:   {sorted(input_models)}\n"
        f"  compiled:  {sorted(compiled_models)}"
    )


def test_infer_example1_no_priors():
    """Step 0 (y_1): Π̄_1 = {}, so S_1 = full domain."""
    cs = parse_constraints(EXAMPLE1_TEXT)
    cc = compile_constraints(cs)
    cf = ClauseFeasibility(cc)
    assert cf.feasible_set(0, {}) == frozenset({0, 1, 2})


def test_infer_example1_step_y2_under_a1_eq_0():
    """y_1=0 (yellow): Π̄_2 clause forces y_2 ∈ {0}."""
    cf = ClauseFeasibility(compile_constraints(parse_constraints(EXAMPLE1_TEXT)))
    assert cf.feasible_set(1, {0: 0}) == frozenset({0})


def test_infer_example1_step_y2_under_a1_eq_1():
    """y_1=1 (red): clause satisfied by y_1 literal -> S_2 unrestricted."""
    cf = ClauseFeasibility(compile_constraints(parse_constraints(EXAMPLE1_TEXT)))
    assert cf.feasible_set(1, {0: 1}) == frozenset({0, 1, 2})


def test_infer_example1_step_y3_circle_prior():
    """y_2=0 (circle): Π̄_3 clause satisfied -> S_3 unrestricted."""
    cf = ClauseFeasibility(compile_constraints(parse_constraints(EXAMPLE1_TEXT)))
    assert cf.feasible_set(2, {0: 1, 1: 0}) == frozenset({0, 1})


def test_infer_example1_step_y3_non_circle_prior():
    """y_2=2 (triangle): y_3 ∈ {0}."""
    cf = ClauseFeasibility(compile_constraints(parse_constraints(EXAMPLE1_TEXT)))
    assert cf.feasible_set(2, {0: 1, 1: 2}) == frozenset({0})


def test_infer_consistency_with_brute_force():
    """For every prior, INFER's feasible set equals the set of values
    extending the prior to a model of Π."""
    cs = parse_constraints(EXAMPLE1_TEXT)
    cc = compile_constraints(cs)
    cf = ClauseFeasibility(cc)

    input_models = set(enumerate_models(cs.clauses, cs.var_domains))

    for step in range(len(cc.ordering)):
        var = cc.ordering[step]
        prior_vars = cc.ordering[:step]
        for prior_values in product(*(range(cc.var_domains[v]) for v in prior_vars)):
            prior = dict(zip(prior_vars, prior_values))
            feasible = cf.feasible_set(step, prior)

            extending = set()
            for model in input_models:
                if all(model[v] == prior[v] for v in prior):
                    extending.add(model[var])

            if not extending:
                continue

            assert feasible == extending, (
                f"step={step} var={var} prior={prior}:\n"
                f"  INFER:       {sorted(feasible)}\n"
                f"  brute-force: {sorted(extending)}"
            )


def test_json_round_trip_example1():
    cc = compile_constraints(parse_constraints(EXAMPLE1_TEXT))
    cc2 = from_json(to_json(cc))
    assert cc2.var_names == cc.var_names
    assert cc2.var_domains == cc.var_domains
    assert cc2.ordering == cc.ordering
    assert cc2.per_var_clauses == cc.per_var_clauses


def test_rectangle_cover_tiles_product_exactly():
    """Every work item is a rectangle; together they partition the full
    cross-product (disjoint + complete) — the invariant the parallel expander
    relies on for its union to equal the serial resolvent set."""
    from resolution import _rectangle_cover

    cases = [
        ([11, 9, 9, 9], 400),
        ([11, 9, 9], 400),
        ([5], 16),
        ([1, 1, 1], 16),
        ([4, 4, 4, 4], 3),
        ([7], 1),
        ([20, 3], 50),
        ([13], 400),
    ]
    for sizes, target in cases:
        rects = _rectangle_cover(sizes, target)
        total = 1
        for s in sizes:
            total *= s
        seen = set()
        for rect in rects:
            ranges = [range(cs, ce) for (cs, ce) in rect]
            ranges += [range(sizes[i]) for i in range(len(rect), len(sizes))]
            for combo in product(*ranges):
                assert combo not in seen, (
                    f"overlap {combo} for sizes={sizes} target={target}"
                )
                seen.add(combo)
        assert len(seen) == total, (
            f"cover gap for sizes={sizes} target={target}: "
            f"{len(seen)} leaves != {total}"
        )


def test_rectangle_cover_fills_cores_on_small_first_group():
    """The bug this fixes: a heavy pattern whose first group is small must still
    yield enough rectangles to saturate the pool — first-group-only chunking
    capped it at the first group's size (~11)."""
    from resolution import _rectangle_cover

    rects = _rectangle_cover([11, 9, 9, 9, 9, 9, 9, 9, 9], target=400)
    assert len(rects) >= 100


@pytest.mark.parametrize("seed", range(8))
def test_parallel_multigroup_matches_serial(seed: int, monkeypatch):
    """Multi-group rectangle splitting (SAT_WORKERS>1) yields a clause set
    identical to the single-core serial compile."""
    import random

    from resolution import CompileTooLarge

    rng = random.Random(1000 + seed)
    text_lines = ["var a 4", "var b 4", "var c 4", "var d 4"]
    for _ in range(8):
        k = rng.randint(2, 4)
        chosen_vars = rng.sample(["a", "b", "c", "d"], k)
        lits = []
        for v in chosen_vars:
            n_vals = rng.randint(1, 3)
            vals = rng.sample([0, 1, 2, 3], n_vals)
            lits.append("[" + ",".join(str(x) for x in vals) + "]:" + v)
        text_lines.append(" or ".join(lits))
    text = "\n".join(text_lines)

    def compile_with(workers: str):
        monkeypatch.setenv("SAT_WORKERS", workers)
        try:
            return compile_constraints(parse_constraints(text)), None
        except (Unsatisfiable, CompileTooLarge) as e:
            return None, type(e)

    serial, serial_exc = compile_with("1")
    parallel, parallel_exc = compile_with("4")

    assert serial_exc is parallel_exc, (
        f"seed={seed}: serial raised {serial_exc}, parallel raised {parallel_exc}"
    )
    if serial is None:
        return

    assert len(serial.per_var_clauses) == len(parallel.per_var_clauses)
    for var, (sv, pv) in enumerate(
        zip(serial.per_var_clauses, parallel.per_var_clauses)
    ):
        assert set(sv) == set(pv), (
            f"seed={seed} var={var}: parallel clause set diverged from serial"
        )
def test_conflict_pattern_global_intersection_proves_no_conflicts():
    from resolution import _enumerate_minimal_conflict_patterns

    value_sets = [
        frozenset({0, 1, 2}),
        frozenset({1, 2, 3}),
        frozenset({1, 4}),
    ]
    assert _enumerate_minimal_conflict_patterns(value_sets, cap=5) == []


