"""Offline compilation of the MNIST-Sum SDD constraint.

For N digits, creates (N*10 + 9*N+1) Boolean variables:
  vars 1..(N*10):              N groups of 10 one-hot digit indicators
  vars (N*10+1)..(N*10+9*N+1): one-hot sum indicators (s_0 ... s_{9*N})

The constraint is:
  (exactly_one per digit group)
  & (DP-factored addition relation with sum one-hot baked in)

The addition disjuncts each assert the correct sum bit AND negate all
others, so no separate sum AMO pass is needed.

Model count should be exactly 10^N.
"""

import argparse
import os
import time

from pysdd.sdd import SddManager, Vtree, WmcManager


def exactly_one(mgr, var_indices):
    """Build an SDD encoding exactly-one-true for the given variable indices."""
    # At least one
    clause = mgr.false()
    for vi in var_indices:
        clause = clause | mgr.vars[vi]

    # At most one: for each pair, not both
    for i in range(len(var_indices)):
        for j in range(i + 1, len(var_indices)):
            clause = clause & (~mgr.vars[var_indices[i]] | ~mgr.vars[var_indices[j]])

    return clause


def main():
    parser = argparse.ArgumentParser(description="Compile MNIST-Sum SDD constraint")
    parser.add_argument("--N", type=int, default=2, help="Number of digits")
    args = parser.parse_args()
    N = args.N

    num_digit_vars = N * 10
    max_sum = 9 * N
    num_sum_vars = max_sum + 1
    total_vars = num_digit_vars + num_sum_vars

    print(f"N={N}, total variables={total_vars} "
          f"(digit: {num_digit_vars}, sum: {num_sum_vars})", flush=True)

    t0 = time.time()

    # Create a balanced vtree with variables ordered by group
    var_order = list(range(1, total_vars + 1))
    vtree = Vtree(var_count=total_vars, var_order=var_order, vtree_type="balanced")
    mgr = SddManager(vtree=vtree, auto_gc_and_minimize=False)

    # ── Addition relation using DP ──
    # Bake sum one-hot directly into each disjunct: each asserts the correct
    # sum bit and negates all others.
    #
    # partial_sums[s] = SDD for "digits in positions 1..k sum to s"
    print(f"Building addition relation (DP over {N} digits)...", flush=True)
    partial_sums = {0: mgr.true()}

    for digit_idx in range(N):
        offset = digit_idx * 10  # var index = offset + d + 1
        new_partial = {}
        for s_prev, sdd_prev in partial_sums.items():
            for d in range(10):
                s_new = s_prev + d
                v_d = mgr.vars[offset + d + 1]
                term = sdd_prev & v_d
                if s_new in new_partial:
                    new_partial[s_new] = new_partial[s_new] | term
                else:
                    new_partial[s_new] = term
        partial_sums = new_partial
        print(f"  digit {digit_idx + 1} done — {len(partial_sums)} partial sums",
              flush=True)

    # For each sum value, build: partial_sums[s] & s_var[s] & AND(~s_var[k] for k!=s)
    # Key optimization: pre-build per-sum-value literal conjunctions as tiny SDDs,
    # then conjoin with partial_sums and OR together.
    print(f"Linking {len(partial_sums)} partial sums to sum indicators...",
          flush=True)
    sum_var_base = num_digit_vars + 1

    # Pre-build the "sum-selector" for each target value: s_target & ~s_0 & ... & ~s_36
    # These are pure-literal conjunctions, cheap to build.
    sum_selectors = {}
    for target in range(num_sum_vars):
        sel = mgr.true()
        for k in range(num_sum_vars):
            sv = mgr.vars[sum_var_base + k]
            sel = sel & (sv if k == target else ~sv)
        sum_selectors[target] = sel
    print(f"  sum selectors built", flush=True)

    addition = mgr.false()
    for s_val in sorted(partial_sums.keys()):
        addition = addition | (partial_sums[s_val] & sum_selectors[s_val])
        if (s_val + 1) % 10 == 0:
            print(f"  linked sum {s_val}/{max_sum}", flush=True)
    print(f"  addition relation complete", flush=True)

    # ── One-hot constraints for each digit group (10 vars each — fast) ──
    alpha = addition
    for digit_idx in range(N):
        offset = digit_idx * 10 + 1
        digit_vars = list(range(offset, offset + 10))
        print(f"Building one-hot for digit {digit_idx + 1} "
              f"(vars {digit_vars[0]}-{digit_vars[-1]})...", flush=True)
        alpha = alpha & exactly_one(mgr, digit_vars)

    elapsed = time.time() - t0

    # Save SDD and vtree
    os.makedirs("sdd_files", exist_ok=True)
    sdd_path = f"sdd_files/mnistsum_N{N}.sdd"
    vtree_path = f"sdd_files/mnistsum_N{N}.vtree"
    alpha.save(sdd_path.encode())
    alpha.vtree().save(vtree_path.encode())

    # Model count via weighted model counting (all weights = 1)
    wmc = WmcManager(alpha, log_mode=False)
    for i in range(1, total_vars + 1):
        wmc.set_literal_weight(mgr.literal(i), 1.0)
        wmc.set_literal_weight(mgr.literal(-i), 1.0)
    model_count = wmc.propagate()

    expected = 10 ** N
    print(f"\nCompilation time: {elapsed:.3f}s")
    print(f"SDD node count: {alpha.count()}")
    print(f"SDD size:       {alpha.size()}")
    print(f"Model count:    {int(model_count)} (expected: {expected})")
    print(f"Saved to: {sdd_path}, {vtree_path}")


if __name__ == "__main__":
    main()
