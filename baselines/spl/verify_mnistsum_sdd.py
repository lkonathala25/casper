"""Sanity-check the compiled MNIST-Sum SDD.

Loads the SDD via CircuitMPE and runs three test cases using
generate_tf_ac in log-space:

1. Consistent:   digits (3, 5), sum=8  -> expected ~0.0
2. Inconsistent: digits (3, 5), sum=7  -> expected ~-300
3. Free sum:     digits (3, 5), all sums free -> expected ~0.0
"""

import sys
import os

# Use grids pypsdd (supports log_space in generate_tf_ac)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "grids", "pypsdd"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "grids"))

import torch
from compute_mpe import CircuitMPE

LOG_ZERO = -300.0  # log(0)


def build_litleaves(digit1, digit2, sum_val=None):
    """Build litleaves for a specific digit pair and optional sum.

    Args:
        digit1: int 0-9, the first digit
        digit2: int 0-9, the second digit
        sum_val: int 0-18 to clamp, or None for free sum

    Returns:
        list of [neg_weight, pos_weight] for all 39 variables (batch_size=1)
    """
    litleaves = []

    # Digit 1 indicators (vars 1-10)
    for k in range(10):
        if k == digit1:
            litleaves.append([torch.tensor([0.0]), torch.tensor([0.0])])
        else:
            litleaves.append([torch.tensor([0.0]), torch.tensor([LOG_ZERO])])

    # Digit 2 indicators (vars 11-20)
    for k in range(10):
        if k == digit2:
            litleaves.append([torch.tensor([0.0]), torch.tensor([0.0])])
        else:
            litleaves.append([torch.tensor([0.0]), torch.tensor([LOG_ZERO])])

    # Sum indicators (vars 21-39)
    for s in range(19):
        if sum_val is None:
            # Free: both weights = 0 (log 1)
            litleaves.append([torch.tensor([0.0]), torch.tensor([0.0])])
        elif s == sum_val:
            litleaves.append([torch.tensor([0.0]), torch.tensor([0.0])])
        else:
            litleaves.append([torch.tensor([0.0]), torch.tensor([LOG_ZERO])])

    return litleaves


def main():
    vtree_path = "sdd_files/mnistsum_N2.vtree"
    sdd_path = "sdd_files/mnistsum_N2.sdd"

    assert os.path.exists(vtree_path), f"Missing {vtree_path} — run compile_mnistsum_sdd.py first"
    assert os.path.exists(sdd_path), f"Missing {sdd_path} — run compile_mnistsum_sdd.py first"

    cmpe = CircuitMPE(vtree_path, sdd_path)
    cmpe.beta.num_reps = 1

    all_passed = True

    # Test 1: Consistent — digits (3, 5), sum=8
    litleaves = build_litleaves(3, 5, sum_val=8)
    result = cmpe.get_tf_ac(litleaves, log_space=True)
    val = result.squeeze().item()
    passed = abs(val - 0.0) < 1.0
    status = "PASS" if passed else "FAIL"
    all_passed &= passed
    print(f"Test 1 (consistent: d1=3, d2=5, sum=8):  log WMC = {val:.4f}  [{status}] (expected ~0.0)")

    # Test 2: Inconsistent — digits (3, 5), sum=7
    litleaves = build_litleaves(3, 5, sum_val=7)
    result = cmpe.get_tf_ac(litleaves, log_space=True)
    val = result.squeeze().item()
    passed = val < -100.0
    status = "PASS" if passed else "FAIL"
    all_passed &= passed
    print(f"Test 2 (inconsistent: d1=3, d2=5, sum=7): log WMC = {val:.4f}  [{status}] (expected ~-300)")

    # Test 3: Free sum — digits (3, 5), all sums free
    litleaves = build_litleaves(3, 5, sum_val=None)
    result = cmpe.get_tf_ac(litleaves, log_space=True)
    val = result.squeeze().item()
    passed = abs(val - 0.0) < 1.0
    status = "PASS" if passed else "FAIL"
    all_passed &= passed
    print(f"Test 3 (free sum: d1=3, d2=5, sum=free):  log WMC = {val:.4f}  [{status}] (expected ~0.0)")

    print()
    if all_passed:
        print("All verification checks PASSED.")
    else:
        print("Some checks FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
