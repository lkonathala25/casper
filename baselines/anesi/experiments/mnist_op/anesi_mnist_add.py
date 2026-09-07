import math
from typing import List

import torch
from experiments.mnist_op import MNISTModel


class MNISTAddNModel(MNISTModel):

    def output_dims(self, N: int, y_encoding: str) -> List[int]:
        if y_encoding == "base10":
            # Two N-digit numbers: max sum is 2 * (10^N - 1), which has N+1 digits
            max_val = 2 * (10 ** N - 1)
            n_digits = max(1, math.ceil(math.log10(max_val + 1)))
            return [10] * n_digits
        raise NotImplementedError

    def op(self, *ns) -> torch.Tensor:
        ns = torch.stack(ns, dim=-1)
        return ns.sum(dim=-1)
