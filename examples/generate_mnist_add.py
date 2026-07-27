from __future__ import annotations

import argparse
from pathlib import Path

DIGITS = list(range(10))


def _neq(name: str, value: int, domain: int = 10) -> str:
    """Signed literal for `name != value` — the complement value-set."""
    others = ",".join(str(v) for v in range(domain) if v != value)
    return f"[{others}]:{name}"


def _eq(name: str, value: int) -> str:
    """Signed literal for `name == value`."""
    return f"[{value}]:{name}"


def generate_carry_dsl(n_digits: int) -> str:
    """Signed-clause DSL for N-digit MNIST-Add via the carry-chain decomposition.

    Variables (index k is significance, k=0 is the units digit):
        a_k, b_k   operand digits, domain 10, k = 0..N-1
        s_k        sum digits,     domain 10, k = 0..N   (s_N is the carry-out)
        c_k        carry into position k, domain 2, k = 1..N-1  (c_0 == 0 is
                   folded into position 0's clauses rather than declared)

    Each position contributes the functional relation
        a_k + b_k + c_k == s_k + 10 * c_{k+1}
    written as one implication clause per (a_k, b_k, c_k) tuple per determined
    output: `a_k != va or b_k != vb or c_k != vc or s_k == vs`. The enumeration
    is exhaustive over the tuple space, so the clause set is exactly equivalent
    to the relation — no relaxation.

    The emitted `ordering` (the INFER assignment order) walks positions LSB to
    MSB, so the reverse — the elimination order — sweeps the carry chain from
    the MSB end. Every bucket is then local to one position, which is what keeps
    the compile linear in N.

    Note the layer's tensor layout is MSB-first (`a_preds[:, 0]` is the most
    significant digit); a_k here is the k-th least significant, i.e. layer index
    N-1-k.
    """
    n = n_digits
    lines: list[str] = []

    for k in range(n):
        lines.append(f"var a_{k} 10")
    for k in range(n):
        lines.append(f"var b_{k} 10")
    for k in range(n + 1):
        lines.append(f"var s_{k} 10")
    for k in range(1, n):
        lines.append(f"var c_{k} 2")

    order: list[str] = []
    for k in range(n):
        order += [f"a_{k}", f"b_{k}", f"s_{k}"]
        if k + 1 < n:
            order.append(f"c_{k + 1}")
    order.append(f"s_{n}")

    lines.append("")
    lines.append("ordering " + " ".join(order))
    lines.append("")

    for k in range(n):
        # Carry into position k: variable c_k, except position 0 where it is
        # the constant 0 and contributes no literal.
        cin_values = [0] if k == 0 else [0, 1]
        # Carry out of position k: c_{k+1}, except the top position where the
        # carry-out IS the leading sum digit s_N.
        cout_name = f"s_{n}" if k == n - 1 else f"c_{k + 1}"
        cout_domain = 10 if k == n - 1 else 2

        for va in DIGITS:
            for vb in DIGITS:
                for vc in cin_values:
                    total = va + vb + vc
                    vs, vcout = total % 10, total // 10
                    guard = [_neq(f"a_{k}", va), _neq(f"b_{k}", vb)]
                    if k > 0:
                        guard.append(_neq(f"c_{k}", vc, domain=2))
                    lines.append(" or ".join(guard + [_eq(f"s_{k}", vs)]))
                    lines.append(
                        " or ".join(guard + [_eq(cout_name, vcout)])
                    )

    return "\n".join(lines) + "\n"


def generate_flat_dsl(n_digits: int) -> str:
    """Signed-clause DSL for N-digit MNIST-Add with no carry variables.

    The addition relation is written directly over the 2N operand digits: one
    implication per (A, B) operand pair per output digit. Clause arity is
    2N+1 and the clause count is 100^N * (N+1), so this is only generatable for
    N <= 2. Kept as the baseline that motivates the carry-chain decomposition.
    """
    n = n_digits
    lines: list[str] = []

    for k in range(n):
        lines.append(f"var a_{k} 10")
    for k in range(n):
        lines.append(f"var b_{k} 10")
    for k in range(n + 1):
        lines.append(f"var s_{k} 10")

    order = [f"a_{k}" for k in range(n)] + [f"b_{k}" for k in range(n)]
    order += [f"s_{k}" for k in range(n + 1)]
    lines.append("")
    lines.append("ordering " + " ".join(order))
    lines.append("")

    limit = 10 ** n
    for a_val in range(limit):
        a_dig = [(a_val // 10 ** k) % 10 for k in range(n)]
        for b_val in range(limit):
            b_dig = [(b_val // 10 ** k) % 10 for k in range(n)]
            total = a_val + b_val
            guard = [_neq(f"a_{k}", a_dig[k]) for k in range(n)]
            guard += [_neq(f"b_{k}", b_dig[k]) for k in range(n)]
            for j in range(n + 1):
                lines.append(
                    " or ".join(guard + [_eq(f"s_{j}", (total // 10 ** j) % 10)])
                )

    return "\n".join(lines) + "\n"


def generate_add_dsl(n_digits: int, encoding: str = "carry") -> str:
    """Dispatch to the carry-chain or flat encoding."""
    if n_digits < 1:
        raise ValueError("n_digits must be >= 1")
    if encoding == "carry":
        return generate_carry_dsl(n_digits)
    if encoding == "flat":
        return generate_flat_dsl(n_digits)
    raise ValueError(f"unknown encoding {encoding!r}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate an MNIST-Add constraint file."
    )
    p.add_argument("n_digits", type=int)
    p.add_argument("out_path", type=Path)
    p.add_argument(
        "--encoding",
        choices=("carry", "flat"),
        default="carry",
        help="carry: carry-chain decomposition (default, scales); "
        "flat: direct relation over all 2N digits (N<=2 only).",
    )
    args = p.parse_args()

    dsl = generate_add_dsl(args.n_digits, args.encoding)
    args.out_path.write_text(dsl)
    n_clauses = sum(
        1 for line in dsl.splitlines()
        if line and not line.startswith(("var", "ordering", "#"))
    )
    n_vars = sum(1 for line in dsl.splitlines() if line.startswith("var "))
    print(
        f"Wrote {n_vars} vars / {n_clauses} clauses "
        f"({args.encoding}, N={args.n_digits}) to {args.out_path}"
    )


if __name__ == "__main__":
    main()
