from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
for _sub in ("casper", "tasks/sudoku"):
    if str(ROOT / _sub) not in sys.path:
        sys.path.insert(0, str(ROOT / _sub))

from constraints import Constraints
from local_casper import (
    LocalGroupCasper,
    alldiff_group_constraints,
    extract_assignments,
    group_violations,
)
from sudoku_groups import build_groups

N = 9
N_CELLS = 81
ARTIFACT = ROOT / "constraints" / "sudoku" / "sudoku_group9_compiled.json"

VALID_BOARD = [
    (3 * (r % 3) + r // 3 + c) % 9 for r in range(N) for c in range(N)
]


def clean_probs(board: list[int], peak: float = 0.80) -> torch.Tensor:
    """81x9 scores flattened to (1, 729): `peak` on the true digit, rest uniform."""
    other = (1.0 - peak) / (N - 1)
    t = torch.full((1, N_CELLS, N), other, dtype=torch.float64)
    for cell, val in enumerate(board):
        t[0, cell, val] = peak
    return t.reshape(1, N_CELLS * N)


def corrupt(
    probs: torch.Tensor, cell: int, wrong: int, true: int
) -> torch.Tensor:
    """Put 0.80 on `wrong`, 0.15 on `true`, 0.05/7 across the remaining digits."""
    out = probs.clone().view(1, N_CELLS, N)
    rest = 0.05 / (N - 2)
    out[0, cell, :] = rest
    out[0, cell, wrong] = 0.80
    out[0, cell, true] = 0.15
    return out.reshape(1, N_CELLS * N)


def board_of(probs: torch.Tensor) -> list[int]:
    """Argmax digit per cell."""
    return extract_assignments(probs, N_CELLS, N)[0]


def run_both(probs: torch.Tensor):
    """Run the identical input through the compiled and closed-form paths."""
    groups = build_groups(N)
    compiled = LocalGroupCasper(Constraints.from_file(ARTIFACT), groups, N)
    closed = LocalGroupCasper(alldiff_group_constraints(N), groups, N)
    with torch.no_grad():
        out_c, info_c = compiled(probs)
        out_f, info_f = closed(probs)
    return out_c, info_c, out_f, info_f


def describe_edits(info) -> str:
    """One-line summary of which passes changed which cells."""
    if not info["edits_by_pass"]:
        return "none"
    parts = []
    for e in info["edits_by_pass"]:
        cells = ",".join(str(c) for _b, c in e["cells"])
        parts.append(f"pass {e['pass']}({e['group']}): cells {cells}")
    return " | ".join(parts)


def report_divergence(exp: list[int], got: list[int], probs, label: str) -> None:
    """First mismatching cell plus its feasible set under both paths."""
    bad = [i for i in range(N_CELLS) if exp[i] != got[i]]
    if not bad:
        return
    cell = bad[0]
    r, c = divmod(cell, N)
    print(f"    first divergence at cell {cell} (r{r + 1},c{c + 1}): "
          f"expected {exp[cell] + 1}, got {got[cell] + 1}  [{label}]")
    groups = dict(build_groups(N))
    for gname in ("row", "col", "box"):
        pass
    # feasible set for that cell inside each of its three groups, given the
    # other eight cells of that group as they stand in `got`
    for name, cells in build_groups(N):
        if cell not in cells:
            continue
        pos = cells.index(cell)
        prior = {i: got[cells[i]] for i in range(pos)}
        fc = Constraints.from_file(ARTIFACT).feasible_set(pos, prior)
        ff = alldiff_group_constraints(N).feasible_set(pos, prior)
        print(f"      {name}: pos={pos} prior={{"
              f"{', '.join(f'{k}:{v + 1}' for k, v in prior.items())}}}")
        print(f"        compiled S_i   = {sorted(x + 1 for x in fc)}")
        print(f"        closed-form S_i= {sorted(x + 1 for x in ff)}")


def check(name: str, cond: bool) -> bool:
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def test0() -> bool:
    print("\n=== Test 0 — valid board is untouched ===")
    probs = clean_probs(VALID_BOARD)
    pre = board_of(probs)
    assert pre == VALID_BOARD, "input argmax should be the valid board"

    out_c, info_c, out_f, info_f = run_both(probs)
    post_c, post_f = board_of(out_c), board_of(out_f)
    viol = group_violations(post_c, build_groups(N), N)

    print(f"    edits (compiled):   {describe_edits(info_c)}")
    print(f"    edits (closed-form):{describe_edits(info_f)}")

    ok = True
    ok &= check("output argmax identical to input on all 81 cells", post_c == pre)
    ok &= check("zero violations in output", not viol)
    ok &= check("compiled tensor == from_function tensor",
                torch.equal(out_c, out_f))
    if post_c != pre:
        report_divergence(pre, post_c, probs, "compiled")
    if viol:
        print(f"    violated groups: {viol}")
    return ok


def plant(kind: str) -> tuple[int, int, int]:
    """Pick a cell and a wrong digit that duplicates inside its `kind` group."""
    groups = dict(build_groups(N))
    # choose a cell whose duplicate partner sits EARLIER in the group's order,
    # so the pass reaches the clash with the duplicate already assigned
    for cell in range(N_CELLS):
        r, c = divmod(cell, N)
        gname = {"row": f"row{r + 1}", "col": f"col{c + 1}",
                 "box": f"box{(r // 3) * 3 + c // 3 + 1}"}[kind]
        cells = groups[gname]
        pos = cells.index(cell)
        if pos == 0:
            continue
        true = VALID_BOARD[cell]
        for earlier in cells[:pos]:
            wrong = VALID_BOARD[earlier]
            if wrong != true:
                return cell, wrong, true
    raise RuntimeError("no candidate cell")


def test1(kind: str, label: str) -> bool:
    print(f"\n=== Test 1{label} — planted duplicate within {kind.upper()} ===")
    cell, wrong, true = plant(kind)
    r, c = divmod(cell, N)
    print(f"    corrupting cell {cell} (r{r + 1},c{c + 1}): "
          f"true digit {true + 1} -> wrong digit {wrong + 1} "
          f"(duplicates within its {kind})")

    probs = corrupt(clean_probs(VALID_BOARD), cell, wrong, true)
    pre = board_of(probs)
    out_c, info_c, out_f, info_f = run_both(probs)
    post_c, post_f = board_of(out_c), board_of(out_f)
    viol = group_violations(post_c, build_groups(N), N)

    edited = [i for i in range(N_CELLS) if pre[i] != post_c[i]]
    print(f"    edits (compiled):   {describe_edits(info_c)}")
    print(f"    edits (closed-form):{describe_edits(info_f)}")
    print(f"    cells changed vs input: {edited}")

    ok = True
    ok &= check("output argmax == original valid board (all 81 cells)",
                post_c == VALID_BOARD)
    ok &= check(f"exactly one cell edited, and it is cell {cell}",
                edited == [cell])
    ok &= check("zero violations in output", not viol)
    ok &= check("compiled tensor == from_function tensor",
                torch.equal(out_c, out_f))
    if post_c != VALID_BOARD:
        report_divergence(VALID_BOARD, post_c, probs, "compiled")
    if viol:
        print(f"    violated groups: {viol}")
    return ok


def main() -> None:
    print(f"artifact: {ARTIFACT}")
    print("group order: rows 1-9, then cols 1-9, then boxes 1-9 (27 passes)")

    results = {"test0": test0()}
    if not results["test0"]:
        print("\nHALT: Test 0 failed; not proceeding to Test 1.")
        sys.exit(1)

    for kind, label in (("row", "a"), ("col", "b"), ("box", "c")):
        results[f"test1{label}"] = test1(kind, label)
        if not results[f"test1{label}"]:
            print(f"\nHALT: Test 1{label} failed; stopping.")
            break

    print("\n=== Summary ===")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
