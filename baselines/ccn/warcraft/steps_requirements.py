import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TASK_DIR = _REPO_ROOT / "tasks" / "warcraft"
for _d in (_REPO_ROOT / "casper", _TASK_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from steps_walk import StepsWalkDecoder  # noqa: E402


def steps_atoms(dec):
    """(tick, cell) -> atom index, ticks in order and cells ascending within each D_i."""
    index = {}
    for i, dom in enumerate(dec.enc.domains):
        for v in dom:
            index[(i, v)] = len(index)
    return index


def steps_clauses(dec, index, at_most_one=True):
    """The steps walk as clauses over one-hot tick atoms, each a list of (atom, sign).

    Per tick: at least one cell of D_i, and pairwise at most one. Per move:
    X_i = u implies X_{i+1} lies in the decoder's feasible set, succ(u) cap
    D_{i+1} -- read from the compiled artifact when the decoder was built from one.
    """
    out = []
    for i, dom in enumerate(dec.enc.domains):
        out.append([(index[(i, v)], True) for v in dom])
        if at_most_one:
            for a in range(len(dom)):
                for b in range(a + 1, len(dom)):
                    out.append([(index[(i, dom[a])], False), (index[(i, dom[b])], False)])
    for i in range(dec.enc.horizon):
        for u in dec.enc.domains[i]:
            out.append([(index[(i, u)], False)]
                       + [(index[(i + 1, v)], True) for v in dec.feasible(i + 1, u)])
    return out


def unit_propagate(clauses):
    """Assign every atom forced by a unit clause and simplify, until none remain.

    The pinned endpoints X_0 = s and X_L = t produce unit clauses, which pishield
    does not accept, so they are resolved here. The result is equivalent.
    """
    fixed = {}
    clauses = [list(c) for c in clauses]
    while True:
        unit = next((c[0] for c in clauses if len(c) == 1), None)
        if unit is None:
            return clauses, fixed
        atom, sign = unit
        if fixed.get(atom, sign) != sign:
            raise ValueError(f"atom {atom} forced both ways: steps clauses unsatisfiable")
        fixed[atom] = sign
        nxt = []
        for c in clauses:
            if (atom, sign) in c:
                continue
            c = [lit for lit in c if lit[0] != atom]
            if not c:
                raise ValueError("empty clause after propagation: steps clauses unsatisfiable")
            nxt.append(c)
        clauses = nxt


def write_steps_requirements(grid_size, path, compiled_artifact=None, at_most_one=True):
    """Write the pishield requirements file after unit propagation.

    Returns (n_atoms, n_clauses, atom_of): atom_of maps each (tick, cell) to its
    pishield variable index, or to True/False where propagation fixed it (the
    pinned endpoints and whatever they force). Variables keep tick order.
    """
    dec = StepsWalkDecoder(grid_size, compiled_artifact=compiled_artifact)
    index = steps_atoms(dec)
    clauses, fixed = unit_propagate(steps_clauses(dec, index, at_most_one))
    free = sorted({a for c in clauses for a, _ in c})
    new = {a: k for k, a in enumerate(free)}
    lines = [" or ".join(f"y_{new[a]}" if sign else f"not y_{new[a]}" for a, sign in c)
             for c in clauses]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")
    atom_of = {key: new.get(a, fixed.get(a)) for key, a in index.items()}
    return len(free), len(lines), atom_of
