import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from steps_requirements import write_steps_requirements  # noqa: E402


def fast_compacted(self):
    """Drop every clause that strictly contains another, comparing only clauses that share a literal.

    Replaces pishield's ClausesGroup.compacted, which compares every pair of clauses
    after each elimination. The result is the same set; the strata it leads to were
    checked identical rule for rule at 8x8 (61 strata, 1,901 rules) and 9x9
    (79 strata, 3,105 rules), at 6x and 10x the speed.
    """
    from pishield.propositional_requirements.clauses_group import ClausesGroup
    clauses = list(self.clauses)
    index = defaultdict(list)
    for k, c in enumerate(clauses):
        for lit in c:
            index[lit].append(k)
    keep = []
    for k, c in enumerate(clauses):
        cands = set()
        for lit in c:
            cands.update(index[lit])
        cands.discard(k)
        if not any(len(clauses[j]) <= len(c) and clauses[j].is_subset(c) for j in cands):
            keep.append(c)
    return ClausesGroup(keep)


def shield_paths(grid_size, at_most_one, out_dir=None):
    """Requirements file, compiled strata and atom map for one CCN+ steps shield."""
    out_dir = Path(out_dir) if out_dir else _REPO_ROOT / "constraints" / "warcraft" / "ccn"
    stem = f"warcraft_ccn_steps_{grid_size}x{grid_size}{'_amo' if at_most_one else ''}"
    return (out_dir / f"{stem}.txt", out_dir / f"{stem}_strata.pkl",
            out_dir / f"{stem}_atoms.json")


def main():
    """Compile the CCN+ steps requirements once and save the strata for train_ccn.py."""
    p = argparse.ArgumentParser()
    p.add_argument("--grid-size", type=int, default=12)
    p.add_argument("--steps-artifact", type=str, default=None,
                   help="Compiled steps artifact; defaults to the one for this grid, "
                        "falling back to the generator when none exists.")
    p.add_argument("--at-most-one", action="store_true",
                   help="Add pairwise at-most-one per tick (much slower to compile).")
    p.add_argument("--ordering", default="custom",
                   choices=["custom", "rev-custom", "rev-degree"],
                   help="custom: eliminate the last tick first; rev-custom: the first.")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--native-compaction", action="store_true",
                   help="Use pishield's own all-pairs subsumption (identical strata, "
                        "6-10x slower at 8x8-9x9 and growing).")
    args = p.parse_args()

    N = args.grid_size
    art = Path(args.steps_artifact) if args.steps_artifact else (
        _REPO_ROOT / "constraints" / "warcraft" / "steps_artifacts"
        / f"steps_{N}_trimmed_compiled.json")
    art = art if art.exists() else None
    req, strata_path, atoms_path = shield_paths(N, args.at_most_one, args.out_dir)
    n, m, atom_of = write_steps_requirements(N, req, compiled_artifact=art,
                                             at_most_one=args.at_most_one)
    print(f"requirements: {req.name} atoms={n} clauses={m} from "
          f"{'compiled artifact ' + art.name if art else 'generator'}", flush=True)
    atoms_path.write_text(json.dumps([[i, v, val] for (i, v), val in sorted(atom_of.items())]))

    from pishield.shield_layer import build_shield_layer
    if not args.native_compaction:
        from pishield.propositional_requirements.clauses_group import ClausesGroup
        ClausesGroup.compacted = fast_compacted
    kw = ({"custom_ordering": ",".join(map(str, range(n)))}
          if "custom" in args.ordering else {})
    t = time.time()
    layer = build_shield_layer(requirements_filepath=str(req), num_variables=n,
                               ordering_choice=args.ordering,
                               requirements_type="propositional", **kw)
    dt = time.time() - t
    with open(strata_path, "wb") as f:
        pickle.dump(layer.stratified_constraints, f)
    print(f"RESULT grid={N} atoms={n} clauses={m} strata={len(layer.strata)} "
          f"compile_s={dt:.1f} compaction={'native' if args.native_compaction else 'indexed'} "
          f"-> {strata_path.name}", flush=True)


if __name__ == "__main__":
    main()
