from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "grids", HERE.parent / "grids" / "pypsdd"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import warcraft_shortest_path.trainers as spl  # noqa: E402
from compute_mpe import CircuitMPE  # noqa: E402

from layered_circuit import LayeredCircuit  # noqa: E402
from steps_sdd import DEFAULT_OUT, build, canonical_edge_ids, load_moves  # noqa: E402
from steps_spl import StepsSplAdapter  # noqa: E402


def e2i_permutation(g: int = 12) -> np.ndarray:
    """perm[j] = canonical edge index of SPL variable j + 1, read from SPL's e2i.pt."""
    e2i = spl.e2i.detach().cpu().numpy() - 1
    ids = canonical_edge_ids(g)
    return np.array([ids[(min(a, b), max(a, b))] for a, b in e2i])


class StepsFFCircuitTrainer(spl.FFCircuitTrainer):
    """SPL's FFCircuit trainer over the steps encoding, without editing SPL's code.

    The base class is constructed exactly as SPL does it (its own path circuit,
    gate and optimizer); only forward_pass is replaced, to score the steps
    circuit instead. The data is handed to SPL in the edge order its e2i table
    and metrics assume, and predictions are returned in that order too.

    SPL_ENGINE=pypsdd evaluates the steps circuit with pypsdd, node by node as
    SPL's own code does; the default, layered, computes the same values one
    circuit depth at a time (tests/test_spl_steps.py checks they agree).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        g = self.grid_dim
        if not (DEFAULT_OUT / f"steps_{g}.sdd").exists():
            print(f"Compiling steps SDD for {g}x{g}: {build(g)}")
        self.cm = CircuitMPE(str(DEFAULT_OUT / f"steps_{g}.vtree"), str(DEFAULT_OUT / f"steps_{g}.sdd"))
        self.cm.beta.num_reps = self.num_reps
        self.adapter = StepsSplAdapter(load_moves(g))
        self.engine_name = os.environ.get("SPL_ENGINE", "layered")
        self.engine = LayeredCircuit(self.cm.beta, self.adapter.sm.num_vars) \
            if self.engine_name == "layered" else None
        self.perm = e2i_permutation(g)
        for it in (self.train_iterator, self.test_iterator):
            it.array["labels"] = it.array["labels"][:, self.perm]
        print(f"StepsFFCircuit: engine={self.engine_name}, labels reordered to SPL's e2i order")

    def to_canonical(self, labels):
        """(B, E) labels in SPL's e2i order -> the canonical order the steps adapter uses."""
        out = torch.zeros_like(labels)
        out[:, torch.as_tensor(self.perm, device=labels.device)] = labels
        return out

    def forward_pass(self, input, label, train, i):

        # Get embedding
        output = self.model(input).view(input.size(0), 264+1, self.num_reps)
        output, mixing = output.split((264, 1), dim=1)
        mixing = mixing.squeeze(1).log_softmax(dim=1)
        self.cm.beta.mixing = mixing

        # SPL's per-edge log-probabilities, moved from e2i order to the canonical order
        logprobs = F.logsigmoid(output).clamp(max=-1e-7)
        logprobs = self.to_canonical(logprobs)
        neg, pos = self.adapter.lit_tensors(logprobs)
        canonical_label = self.to_canonical(label)

        if train:
            target = self.adapter.edge_labels_to_bits(canonical_label)
            if self.engine is not None:
                loss = self.engine.cross_entropy(neg, pos, target, mixing).mean()
            else:
                loss = self.cm.ff_cross_entropy(target, [[n, p] for n, p in zip(neg.unbind(1), pos.unbind(1))]).mean()
            print(f"Loss at iter {i}: {loss.item()}")
            return loss

        bsz = label.size()[0]
        flat_target = label.view(bsz, -1)
        if self.engine is not None:
            bits = self.engine.mpe_bits(neg, pos, mixing)
        else:
            self.cm.parameterize_ff([[n, p] for n, p in zip(neg.unbind(1), pos.unbind(1))])
            bits = (self.cm.get_mpe_inst(bsz) > 0).long()
        edges, _ = self.adapter.bits_to_edges(bits)
        suggested_path = edges[:, torch.as_tensor(self.perm, device=edges.device)].to(flat_target.dtype)

        accuracy = (suggested_path * flat_target).sum().item() / flat_target.sum().item()
        return accuracy, {"vertex_costs": None, "suggested_path": suggested_path}
