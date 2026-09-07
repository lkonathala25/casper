"""Offline compilation of the MNIST-Add SDD constraint.

For N digits per operand, MNIST-Add models:
  a = sum_{i=0..N-1} d_a[i] * 10^(N-1-i)
  b = sum_{i=0..N-1} d_b[i] * 10^(N-1-i)
  sum = a + b      ∈ {0 .. 2*(10^N - 1)}

Variables (2N digits total, 2N*10 + (2*10^N - 1) Boolean vars):
  vars 1 .. 2N*10                 : 2N groups of 10 one-hot digit indicators
                                    (operand a's digits MSB→LSB, then b's)
  vars 2N*10+1 .. end             : one-hot sum indicators (s_0 ... s_max)

The constraint is:
  (exactly_one per digit group)
  & (DP-factored weighted-addition relation with sum one-hot baked in)

Each digit contributes its place value (10^k) to the running partial sum.
Model count should be exactly 10^(2N).
"""

import argparse
import os
import time

from pysdd.sdd import SddManager, Vtree, WmcManager


def exactly_one(mgr, var_indices):
    """Build an SDD encoding exactly-one-true for the given variable indices."""
    clause = mgr.false()
    for vi in var_indices:
        clause = clause | mgr.vars[vi]
    for i in range(len(var_indices)):
        for j in range(i + 1, len(var_indices)):
            clause = clause & (~mgr.vars[var_indices[i]] | ~mgr.vars[var_indices[j]])
    return clause


def main():
    parser = argparse.ArgumentParser(description="Compile MNIST-Add SDD constraint")
    parser.add_argument("--N", type=int, default=2, help="Digits per operand (a,b)")
    args = parser.parse_args()
    N = args.N

    num_digits = 2 * N
    num_digit_vars = num_digits * 10
    max_operand = 10 ** N - 1
    max_sum = 2 * max_operand
    num_sum_vars = max_sum + 1
    total_vars = num_digit_vars + num_sum_vars

    # Place value of each digit position (MSB-first within each operand)
    weights = [10 ** (N - 1 - (i % N)) for i in range(num_digits)]

    print(f"N={N} (digits per operand), total digits={num_digits}", flush=True)
    print(f"Total variables={total_vars} "
          f"(digit: {num_digit_vars}, sum: {num_sum_vars} for sums 0..{max_sum})",
          flush=True)
    print(f"Place-value weights: {weights}", flush=True)

    t0 = time.time()

    var_order = list(range(1, total_vars + 1))
    vtree = Vtree(var_count=total_vars, var_order=var_order, vtree_type="balanced")
    mgr = SddManager(vtree=vtree, auto_gc_and_minimize=False)

    # ── Weighted addition relation via DP ──
    # partial_sums[s] = SDD for "digits in positions 1..k contribute weighted sum s"
    print(f"Building weighted-addition relation (DP over {num_digits} digits)...",
          flush=True)
    partial_sums = {0: mgr.true()}

    for digit_idx in range(num_digits):
        offset = digit_idx * 10
        w = weights[digit_idx]
        new_partial = {}
        for s_prev, sdd_prev in partial_sums.items():
            for d in range(10):
                s_new = s_prev + d * w
                v_d = mgr.vars[offset + d + 1]
                term = sdd_prev & v_d
                if s_new in new_partial:
                    new_partial[s_new] = new_partial[s_new] | term
                else:
                    new_partial[s_new] = term
        partial_sums = new_partial
        print(f"  digit {digit_idx + 1}/{num_digits} (weight {w}) done — "
              f"{len(partial_sums)} partial sums", flush=True)

    # Pre-build per-target-sum literal conjunctions over the sum indicators.
    print(f"Linking {len(partial_sums)} partial sums to sum indicators...",
          flush=True)
    sum_var_base = num_digit_vars + 1

    sum_selectors = {}
    for target in range(num_sum_vars):
        sel = mgr.true()
        for k in range(num_sum_vars):
            sv = mgr.vars[sum_var_base + k]
            sel = sel & (sv if k == target else ~sv)
        sum_selectors[target] = sel
    print(f"  sum selectors built", flush=True)

    addition = mgr.false()
    progress_step = max(1, len(partial_sums) // 10)
    for i, s_val in enumerate(sorted(partial_sums.keys())):
        addition = addition | (partial_sums[s_val] & sum_selectors[s_val])
        if (i + 1) % progress_step == 0:
            print(f"  linked {i + 1}/{len(partial_sums)} partial sums",
                  flush=True)
    print(f"  addition relation complete", flush=True)

    # ── One-hot constraints for each digit group ──
    alpha = addition
    for digit_idx in range(num_digits):
        offset = digit_idx * 10 + 1
        digit_vars = list(range(offset, offset + 10))
        print(f"Building one-hot for digit {digit_idx + 1} "
              f"(vars {digit_vars[0]}-{digit_vars[-1]})...", flush=True)
        alpha = alpha & exactly_one(mgr, digit_vars)

    elapsed = time.time() - t0

    os.makedirs("sdd_files", exist_ok=True)
    sdd_path = f"sdd_files/mnistadd_N{N}.sdd"
    vtree_path = f"sdd_files/mnistadd_N{N}.vtree"
    alpha.save(sdd_path.encode())
    alpha.vtree().save(vtree_path.encode())

    wmc = WmcManager(alpha, log_mode=False)
    for i in range(1, total_vars + 1):
        wmc.set_literal_weight(mgr.literal(i), 1.0)
        wmc.set_literal_weight(mgr.literal(-i), 1.0)
    model_count = wmc.propagate()

    expected = 10 ** num_digits
    print(f"\nCompilation time: {elapsed:.3f}s")
    print(f"SDD node count: {alpha.count()}")
    print(f"SDD size:       {alpha.size()}")
    print(f"Model count:    {int(model_count)} (expected: {expected})")
    print(f"Saved to: {sdd_path}, {vtree_path}")


if __name__ == "__main__":
    main()
