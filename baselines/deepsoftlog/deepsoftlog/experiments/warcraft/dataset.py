"""DSL-compatible warcraft dataset (grid size set by set_grid, default 12x12).

Wraps the shared tasks/warcraft/data.WarcraftDataset so each
__getitem__ returns a Query(Expr("valid", TensorTerm(img))). Keeps the
underlying numpy arrays accessible for raw-task-accuracy / endpoint-
agreement diagnostics.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# The shared Warcraft task modules (categories, dataset, loss, metrics) live in
# tasks/warcraft of this repo, alongside CASPER.
_REPO_ROOT = Path(__file__).resolve().parents[5]
_TASK_DIR = _REPO_ROOT / "tasks" / "warcraft"
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))

from node_categories import build_node_categories  # noqa: E402
from data import WarcraftDataset as _RawWarcraftDataset  # noqa: E402

from deepsoftlog.algebraic_prover.terms.expression import Expr  # noqa: E402
from deepsoftlog.data.dataset import Dataset  # noqa: E402
from deepsoftlog.data.query import Query  # noqa: E402
from deepsoftlog.logic.soft_term import TensorTerm  # noqa: E402


GRID = 12
ALPHABET = "full"
DATA_DIR_DEFAULT = _REPO_ROOT / "data" / "warcraft" / "12x12" / "preprocessed_4conn"


def set_grid(grid_size):
    """Set the grid size used by the dataset and category setup, and its default data dir."""
    global GRID, DATA_DIR_DEFAULT
    GRID = int(grid_size)
    DATA_DIR_DEFAULT = (_REPO_ROOT / "data" / "warcraft" / f"{GRID}x{GRID}"
                        / "preprocessed_4conn")


def get_node_setup(grid_size=None):
    """(node_info, var_ranges, node_order, total_dim)."""
    return build_node_categories(GRID if grid_size is None else grid_size,
                                 alphabet=ALPHABET)


class WarcraftQueryDataset(Dataset):
    """Returns either:
        - plain Query(Expr("valid", TensorTerm(img)))             [with_labels=False]
        - (Query, node_label_tensor)                              [with_labels=True]

    The label-bearing variant is consumed by WarcraftTrainer.train_epoch,
    which needs per-node CE alongside the prover NLL. Trainer.eval() is
    only ever called on a plain-Query variant (val/test loaders).

    Also exposes:
      - .raw_dataset: underlying WarcraftDataset (PyTorch-style indexing)
      - .node_labels[i] / .edge_labels[i]: ground-truth tensors for image i
    """

    def __init__(self, split, data_dir=None, node_info=None, node_order=None,
                 with_labels=False):
        if data_dir is None:
            data_dir = DATA_DIR_DEFAULT
        if node_info is None or node_order is None:
            node_info, _, node_order, _ = get_node_setup()
        self.raw_dataset = _RawWarcraftDataset(data_dir, split, node_info, node_order)
        self.split = split
        self.with_labels = with_labels

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        img, node_label, _edge_label = self.raw_dataset[index]
        # output_ind=() because valid/1 has no "output" slot --
        # supervision is just P(valid)=1.
        q = Query(Expr("valid", TensorTerm(img)), p=1.0, output_ind=())
        if self.with_labels:
            return q, node_label
        return q

    def get_vocabulary(self):
        """Only the constant '1' is needed; sample one item to satisfy
        Dataset.get_vocabulary without scanning all 10k images."""
        from deepsoftlog.parser.vocabulary import Vocabulary
        item = self[0]
        q = item[0] if isinstance(item, tuple) else item
        return Vocabulary().add_all([q.query])

    @property
    def edge_labels(self):
        return self.raw_dataset.edge_labels

    @property
    def node_labels(self):
        return self.raw_dataset.node_labels

    @property
    def maps(self):
        return self.raw_dataset.maps
