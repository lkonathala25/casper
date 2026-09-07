"""Sanity-check the compiled MNIST-Add SDD (N=2 by default).

For N=2 the operands are a = 10*d1 + d2 and b = 10*d3 + d4, so e.g.
  d=[5,3,4,4]  →  a=53, b=44, sum=97.

Loads the SDD via CircuitMPE and runs three test cases using
generate_tf_ac in log-space:

1. Consistent:   d=[5,3,4,4], sum=97  -> expected ~0.0
2. Inconsistent: d=[5,3,4,4], sum=98  -> expected ~-300
3. Free sum:     d=[5,3,4,4], all sums free -> expected ~0.0
"""

import argparse
import os
import sys

# Use grids pypsdd (supports log_space in generate_tf_ac)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "grids", "pypsdd"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "grids"))

import torch
from compute_mpe import CircuitMPE

LOG_ZERO = -300.0


def build_litleaves(digits, N, sum_val=None):
    """Build litleaves for a specific 2N-digit assignment and optional sum.

    Args:
        digits: list/tuple of length 2N, each in 0..9 (operand a's digits
                MSB→LSB, then operand b's digits)
        N: digits per operand
        sum_val: int 0..2*(10^N - 1) to clamp, or None for free sum

    Returns:
        list of [neg_weight, pos_weight] for all variables (batch_size=1)
    """
    num_digits = 2 * N
    assert len(digits) == num_digits
    max_sum = 2 * (10 ** N - 1)
    num_sum_vars = max_sum + 1

    litleaves = []

    # Digit indicators: num_digits groups of 10 vars each
    for j in range(num_digits):
        for k in range(10):
            if k == digits[j]:
                litleaves.append([torch.tensor([0.0]), torch.tensor([0.0])])
            else:
                litleaves.append([torch.tensor([0.0]), torch.tensor([LOG_ZERO])])

    # Sum indicators
    for s in range(num_sum_vars):
        if sum_val is None:
            litleaves.append([torch.tensor([0.0]), torch.tensor([0.0])])
        elif s == sum_val:
            litleaves.append([torch.tensor([0.0]), torch.tensor([0.0])])
        else:
            litleaves.append([torch.tensor([0.0]), torch.tensor([LOG_ZERO])])

    return litleaves


def true_sum(digits, N):
    a = sum(digits[i] * 10 ** (N - 1 - i) for i in range(N))
    b = sum(digits[N + i] * 10 ** (N - 1 - i) for i in range(N))
    return a + b


def main():
    parser = argparse.ArgumentParser(description="Verify MNIST-Add SDD")
    parser.add_argument("--N", type=int, default=2)
    args = parser.parse_args()
    N = args.N

    vtree_path = f"sdd_files/mnistadd_N{N}.vtree"
    sdd_path = f"sdd_files/mnistadd_N{N}.sdd"
    assert os.path.exists(vtree_path), f"Missing {vtree_path} — run compile_mnistadd_sdd.py first"
    assert os.path.exists(sdd_path), f"Missing {sdd_path} — run compile_mnistadd_sdd.py first"

    cmpe = CircuitMPE(vtree_path, sdd_path)
    cmpe.beta.num_reps = 1

    # Pick a sample assignment for N=2: d=[5,3,4,4] → 53+44 = 97
    if N == 2:
        digits = [5, 3, 4, 4]
    else:
        # Generic: alternate digits, e.g. for N=3 → d=[1,2,3,4,5,6] → 123+456 = 579
        digits = [(i % 9) + 1 for i in range(2 * N)]
    correct_sum = true_sum(digits, N)
    wrong_sum = correct_sum + 1

    a = sum(digits[i] * 10 ** (N - 1 - i) for i in range(N))
    b = sum(digits[N + i] * 10 ** (N - 1 - i) for i in range(N))
    print(f"Test assignment: digits={digits}  →  a={a}, b={b}, a+b={correct_sum}")

    all_passed = True

    # Test 1: Consistent
    litleaves = build_litleaves(digits, N, sum_val=correct_sum)
    val = cmpe.get_tf_ac(litleaves, log_space=True).squeeze().item()
    passed = abs(val - 0.0) < 1.0
    all_passed &= passed
    print(f"Test 1 (consistent: sum={correct_sum}):    log WMC = {val:.4f}  "
          f"[{'PASS' if passed else 'FAIL'}] (expected ~0.0)")

    # Test 2: Inconsistent
    litleaves = build_litleaves(digits, N, sum_val=wrong_sum)
    val = cmpe.get_tf_ac(litleaves, log_space=True).squeeze().item()
    passed = val < -100.0
    all_passed &= passed
    print(f"Test 2 (inconsistent: sum={wrong_sum}):  log WMC = {val:.4f}  "
          f"[{'PASS' if passed else 'FAIL'}] (expected ~-300)")

    # Test 3: Free sum
    litleaves = build_litleaves(digits, N, sum_val=None)
    val = cmpe.get_tf_ac(litleaves, log_space=True).squeeze().item()
    passed = abs(val - 0.0) < 1.0
    all_passed &= passed
    print(f"Test 3 (free sum):                  log WMC = {val:.4f}  "
          f"[{'PASS' if passed else 'FAIL'}] (expected ~0.0)")

    print()
    if all_passed:
        print("All verification checks PASSED.")
    else:
        print("Some checks FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
